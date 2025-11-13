
import ray
import numpy as np
import re
import pprint

class CurriculumDatasetWrapper:
    def __init__(self, dataset, initial_portion=0.0, prompt_key='prompt', target_key='answer', seperator=None, **kwargs):
        self.dataset = dataset  # Keep as HF Dataset
        self.ground_truth_portion_dist = initial_portion  # Start with a small proportion
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.global_step = 0
        self.seperator = seperator
        # self.portions = []

    def __getitem__(self, idx):
        """Retrieves a dataset sample with a dynamically adjusted reasoning portion."""
        sample = self.dataset[idx]  # Directly access HF dataset  
        sample['extra_info']['index'] = idx
        if sample['extra_info'].get('partial_answer', None) is None:
            portion = self._compute_portion_for_sample(idx)
            return self._apply_portion_to_sample_(sample, portion)
        else:
            return sample

    def _compute_portion_for_sample(self, idx):
        portion = self.sample_portion(seed=self.return_seed(idx), size=1)[0] # Get current difficulty proportion
        return portion

    def sample_portion(self, seed=42, size=1):
        """Returns the current proportion of ground-truth CoT to use."""
        if callable(self.ground_truth_portion_dist):
            return self.ground_truth_portion_dist(seed, size)
        elif isinstance(self.ground_truth_portion_dist, float):
            return [self.ground_truth_portion_dist]
        else:
            raise ValueError("ground_truth_portion_dist should be a float or a callable")

    def _apply_portion_to_sample_(self, sample, portion):
        # Modify reasoning exposure based on `portion`
        sample['extra_info']['portion'] = portion
        if portion > 0.0:
            reasoning_steps = sample.get(self.target_key, '')
            if self.seperator == 'char':
                word_list = list(reasoning_steps)
                cut_answer = ''.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer
                completion = ''.join(word_list[int(len(word_list) * portion):])  # Remaining CoT answer
            elif self.seperator is not None:
                word_list = reasoning_steps.split(self.seperator)
                cut_answer = self.seperator.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer
                completion = self.seperator.join(word_list[int(len(word_list) * portion):])  # Remaining CoT answer
            elif self.seperator is None:
                matches = list(re.finditer(r'\S+\s*', reasoning_steps))
                cut_index = int(len(matches) * portion)
                cut_answer = ''.join(m.group(0) for m in matches[:cut_index])
                completion = ''.join(m.group(0) for m in matches[cut_index:])
                # word_list = reasoning_steps.split()
                # cut_answer = ' '.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer
                # completion = ' '.join(word_list[int(len(word_list) * portion):])  # Remaining CoT answer
            
            if cut_answer != '':
                sample['extra_info']['partial_answer'] = cut_answer
                sample['extra_info']['completion'] = completion
        return sample

    def sync_with_all_datasets(self, state=None):
        self.global_step = state.get('global_step', self.global_step)

    def return_seed(self, idx):
        return self.global_step + 1024 * idx

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """Iterates over the dataset."""
        for i in range(len(self.dataset)):
            yield self[i]

    

import os
import numpy as np
class PerSampleCurriculumDatasetWrapper(CurriculumDatasetWrapper):
    def __init__(self, dataset, initial_portion=0.0, prompt_key='prompt', target_key='answer',
                 reward_threshold=0.5, seperator=None, zero_prob=0, max_ratio=0.9, min_ratio=0.0):
        super().__init__(dataset, initial_portion, prompt_key, target_key, seperator=seperator)
        self.reward_threshold = reward_threshold
        self.max_ratio_allowed = max_ratio
        self.min_mean_ratio = min_ratio
        self.max_mean_ratio = max_ratio
        self.max_per_sample_ratio = [self.max_mean_ratio for _ in range(len(self.dataset))]
        self.attempted_ratio_list = [[] for _ in range(len(self.dataset))]  # Store attempted ratios for each sample
        self.global_step = 0
        self.zero_prob = zero_prob
        
        self.wrote_file_step = 0
        self.this_process_tried_portions = []  # for debugging

        self.counter = 0

    def _compute_portion_for_sample(self, idx):
        # self.counter += 1
        # if self.counter % 256 ==0:
        #     self.save_portions_to_file()
        upper_bound = self.max_per_sample_ratio[idx]
        lower_bound = 0.0 # because we never go back/ Ma be aghab bar nemigardim
        attempted_ratio_list = self.attempted_ratio_list[idx]

        if attempted_ratio_list:
            last_gen_portion = attempted_ratio_list[-1]['portion']
            last_gen_avg_reward = np.mean(attempted_ratio_list[-1]['reward'])
            if last_gen_avg_reward < self.reward_threshold:
                lower_bound = last_gen_portion
            else:
                upper_bound = last_gen_portion
                # lower_bound = max(last_gen_portion-0.1, 0.0)
        else:
            # moving average of the min and max ratio
            lower_bound = self.min_mean_ratio
            upper_bound = self.max_mean_ratio

        # if upper_bound <= 0:
        #     print("Havij", f"\n{self.max_per_sample_ratio[idx]}, {len(self.max_per_sample_ratio)}, {(self.min_mean_ratio, self.max_mean_ratio)}\n" , f'idx={idx},\n lower_bound={lower_bound},\n upper_bound={upper_bound},\n last_gen_portion={last_gen_portion},\n last_gen_reward={last_gen_avg_reward}, attempted_ratio_list={attempted_ratio_list},\n', self.max_per_sample_ratio, flush=True)
        # if lower_bound > upper_bound:
        #     print("CHECK THIS", f"\n{self.max_per_sample_ratio[idx]}, {len(self.max_per_sample_ratio)}, {(self.min_mean_ratio, self.max_mean_ratio)}\n", f'idx={idx},\n lower_bound={lower_bound},\n upper_bound={upper_bound},\n last_gen_portion={last_gen_portion},\n last_gen_reward={last_gen_avg_reward}, attempted_ratio_list={attempted_ratio_list},\n', self.max_per_sample_ratio, flush=True)
        #     raise Exception(f'idx={idx},\n lower_bound={lower_bound},\n upper_bound={upper_bound},\n last_gen_portion={last_gen_portion},\n last_gen_reward={last_gen_avg_reward}, attempted_ratio_list={attempted_ratio_list},\n{self.max_per_sample_ratio}')
        
        portion = self._sample_ratio_with_seed(seed=self.return_seed(idx), size=1, lower_bound=lower_bound, upper_bound=upper_bound)[0]
        # self.this_process_tried_portions.append((idx, portion))
        return portion
    
    def sync_with_all_datasets(self, state=None):
        # Sync the ratio actor with all datasets
        # self.save_portions_to_file()
        self.max_per_sample_ratio = state['max_per_sample_ratio']
        self.min_mean_ratio = state['mean_min_ratio']
        self.max_mean_ratio = state['mean_max_ratio'] # because we never go back/ Ma be aghab bar nemigardim
        self.attempted_ratio_list = state['attempted_ratios_list']
        self.global_step = state['global_step']

    # Sample new portion from updated uniform distribution
    def _sample_ratio_with_seed(self, seed=42, size=1, lower_bound=0.0, upper_bound=1.0):
        if self.zero_prob == 0.0:
            return np.random.default_rng(seed).uniform(low=lower_bound, high=upper_bound, size=size)
        elif self.zero_prob == "linear":
            rng = np.random.default_rng(seed)
            zero = rng.uniform(low=0, high=1, size=1) >= upper_bound
            if zero:
                return [0.0]
            else:
                return rng.uniform(low=lower_bound, high=upper_bound, size=size)
        elif self.zero_prob == "uniform":
            return np.random.default_rng(seed).uniform(low=0, high=0.9, size=size)
        elif self.zero_prob == "uniform-linear":
            rng = np.random.default_rng(seed)
            zero = rng.uniform(low=0, high=1, size=1) >= 0.5
            if zero:
                return [0.0]
            else:
                return rng.uniform(low=0, high=0.9, size=size)
        else:
            rng = np.random.default_rng(seed)
            zero = rng.uniform(0, 1, 1) <= float(self.zero_prob)
            if zero:
                return [0.0]
            else:
                return rng.uniform(low=lower_bound, high=upper_bound, size=size)

    def set_portion(self, portion):
        raise NotImplementedError("Portion setting is not supported for per-sample curriculum learning.")

    # for debugging if all processes are getting the same portions/values
    # def save_portions_to_file(self,):
    #     """Write attempted sample ratios and bounds to file for debugging."""
    #     pid = os.getpid()
    #     os.makedirs("/mnt/task_wrapper/user_output/artifacts/debugging", exist_ok=True)
    #     os.makedirs(f"/mnt/task_wrapper/user_output/artifacts/debugging/portion_pid{pid}", exist_ok=True)
    #     path = f'/mnt/task_wrapper/user_output/artifacts/debugging/portion_pid{pid}/step{self.wrote_file_step}.txt'

    #     with open(path, 'w') as f:
    #         f.write(f"=== Debug info for PID {pid} ===\n\n")
    #         f.write(f"Global step: {self.global_step}\n")
    #         f.write(f"Min mean ratio: {self.min_mean_ratio}\n")
    #         f.write(f"Max mean ratio: {self.max_mean_ratio}\n\n")

    #         f.write("Sampled portions in this process:\n")
    #         for idx, portion in self.this_process_tried_portions:
    #             f.write(f"  - idx: {idx}, portion: {portion:.4f}\n")

    #         f.write("\nPer-sample max ratios and attempts:\n")
    #         for i in range(len(self.dataset)):
    #             f.write(f"\nidx {i}:\n")
    #             f.write(f"  max_per_sample_ratio: {self.max_per_sample_ratio[i]:.4f}\n")
    #             for entry in self.attempted_ratio_list[i]:
    #                 f.write(f"    portion: {entry['portion']}, rewards: {entry['reward']}\n")

    #     # Reset logging
    #     self.wrote_file_step += 1
    #     self.this_process_tried_portions = []



@ray.remote
class RatioAttemptsVariablesActor:
    def __init__(self, dataset_length, min_ratio=0.0, max_ratio=0.9, moving_avg_alpha=0.8, reward_threshold=0.5):
        self.attempted_ratios_list = [[] for _ in range(dataset_length)]
        self.newly_added_ids = set()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.reward_threshold = reward_threshold
        self.moving_avg_alpha = moving_avg_alpha
        self.global_step = 0
        self.mean_min_ratio = min_ratio
        self.mean_max_ratio = max_ratio        
        self.max_per_sample_ratio = [self.max_ratio for _ in range(dataset_length)]

    def update_attempted_ratios(self, gathered_data):
        # pid = os.getpid()
        # pprint.pp(f'Updating attempted ratios in PID {pid} with these samples: {gathered_data} \n \n')
        for ids, portions, rewards in gathered_data:
            for id_, portion, reward in zip(ids, portions, rewards):
                entry_list = self.attempted_ratios_list[id_]
                if not entry_list or entry_list[-1]['portion'] != portion:
                    entry_list.append({'portion': portion, 'reward': [reward]})
                else:
                    entry_list[-1]['reward'].append(reward)
                self.newly_added_ids.add(id_)
        return 1

    def update_min_max_avg_ratios(self,):
        # Update global mean bounds using newly added IDs
        avg_macro_upper = 0
        avg_macro_lower = 0
        for id_ in self.newly_added_ids:
            entry = self.attempted_ratios_list[id_][-1]
            avg_reward = np.mean(entry['reward'])
            if avg_reward < self.reward_threshold:
                avg_macro_lower += entry['portion']
                avg_macro_upper += self.max_per_sample_ratio[id_]
            else:
                avg_macro_upper += entry['portion']
                avg_macro_lower += self.min_ratio
                self.max_per_sample_ratio[id_] = entry['portion']        

        count = len(self.newly_added_ids)
        if count > 0:
            self.mean_max_ratio = (
                self.moving_avg_alpha * self.mean_max_ratio +
                (1 - self.moving_avg_alpha) * (avg_macro_upper / count)
            )
            self.mean_min_ratio = (
                self.moving_avg_alpha * self.mean_min_ratio +
                (1 - self.moving_avg_alpha) * (avg_macro_lower / count)
            )

        self.newly_added_ids.clear()
        return 1

    def set_global_step(self, step):
        self.global_step = step
        return 1

    def get_state(self):
        return {
            'attempted_ratios_list': self.attempted_ratios_list,
            'newly_added_ids': self.newly_added_ids,
            'mean_min_ratio': self.mean_min_ratio,
            'mean_max_ratio': self.mean_max_ratio,
            'max_per_sample_ratio': self.max_per_sample_ratio
        }
    
    def set_state(self, states):
        # to avoid read-only error, we need to copy each object
        # breakpoint()
        self.attempted_ratios_list = [list(x) for x in states['attempted_ratios_list']]
        for i in range(len(self.attempted_ratios_list)):
            self.attempted_ratios_list[i] = [{'portion': x['portion'], 'reward': list(x['reward'])} for x in self.attempted_ratios_list[i]]
        self.newly_added_ids = set(states['newly_added_ids'])
        self.max_per_sample_ratio = [x for x in states['max_per_sample_ratio']]
        self.mean_max_ratio = states['mean_max_ratio']
        self.mean_min_ratio = states['mean_min_ratio']
        return 1


