from verl.workers.reward_manager import NaiveRewardManager
from verl import DataProto
from collections import defaultdict
import torch

class NaiveRewardManagerWithPortionLogging(NaiveRewardManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        reward_extra_info['index'] = data.non_tensor_batch['index']

        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch['reward_model'].get('ground_truth', None)
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # if partial answer was provided, we prepend to the response
            if extra_info and extra_info.get('partial_answer', None) is not None:
                response_str = extra_info['partial_answer'] + response_str

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            reward_extra_info['portion'].append(data_item.non_tensor_batch['extra_info'].get('portion', 0.0))
            scores.append(score)
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        # logging the rewards and portions that were used in the batch
        self.trainer.log(data, reward_extra_info)
        # updating the (adaptive) dataset with the attempted ratios and their rewards
        self.trainer.update_datasets_with_ratios(data, scores ,reward_extra_info)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor