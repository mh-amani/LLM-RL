from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import collate_fn
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from omegaconf import OmegaConf, open_dict
import wandb
import numpy as np
import os
import ray
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
import hydra
import pprint


from src.utils.curriculum_dataset_wrapper import RatioAttemptsVariablesActor, \
                                                    CurriculumDatasetWrapper, PerSampleCurriculumDatasetWrapper

class RayPPOTrainerNonParquetteDataset(RayPPOTrainer):
    """
    This class is used to train a PPO model with a non-parquet dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size

        # taking care of the cases where a chat template is not defined. 
        if self.tokenizer.chat_template is None:
            print("No chat template found. Setting a custom one. it should take care of add_generation_prompt")
            self.tokenizer.chat_template = """{% for message in messages -%}
                                            {{ message['role'] }}: {{ message['content'] }}
                                            {% endfor -%}{% if add_generation_prompt -%}
                                            assistant: {% endif %}"""

        print(self.config.data.train_files)
        self.train_dataset = AdaptiveRLHFDataset(type=self.config.data.train_dataset_type, curriculum_config=self.config.data.curriculum_config,
                                                size=self.config.data.train_size,
                                        parquet_files=self.config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        processor=self.processor,
                                        prompt_key=self.config.data.prompt_key,
                                        image_key=self.config.data.get('image_key', 'images'),
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation=self.config.data.get('truncation', 'error'),
                                        filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                        num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'

        self.ratio_actor = RatioAttemptsVariablesActor.remote(dataset_length=len(self.train_dataset.dataframe),
                                                                min_ratio=self.config.data.curriculum_config.get('min_ratio', 0.0),
                                                                max_ratio=self.config.data.curriculum_config.get('max_ratio', 0.9),
                                                                moving_avg_alpha=0.8,
                                                                reward_threshold=0.5)

        if self.config.data.get('sampler', None) is not None:
            sampler = hydra.utils.instantiate(self.config.data.sampler, dataset_size=len(self.train_dataset),
                                              attempted_ratio_list=self.train_dataset.dataframe.attempted_ratio_list,
                                              epsilon=0.01, easy_floor=0.02)
        elif self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                batch_size=self.config.data.get('gen_batch_size',
                                                                                self.config.data.train_batch_size),
                                                num_workers=8,
                                                drop_last=True,
                                                collate_fn=collate_fn,
                                                sampler=sampler)

        self.val_dataset = AdaptiveRLHFDataset(type='base', curriculum_config=self.config.data.curriculum_config,
                                               size=self.config.data.test_size,
                                    parquet_files=self.config.data.val_files,
                                    tokenizer=self.tokenizer,
                                    processor=self.processor,
                                    prompt_key=self.config.data.prompt_key,
                                    image_key=self.config.data.get('image_key', 'images'),
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation=self.config.data.get('truncation', 'error'),
                                    filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                    num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
    

    def _save_checkpoint(self):
        super()._save_checkpoint()
        # save the dataset state
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        ratio_actor_local_path = os.path.join(local_global_step_folder, 'ratio_actor.pt')
        ratio_actor_state_dict = ray.get(self.ratio_actor.get_state.remote())
        torch.save(ratio_actor_state_dict, ratio_actor_local_path)
    

    def _load_checkpoint(self):
        super()._load_checkpoint()
        # load the dataset state
        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
        if global_step_folder is None:
            return
        ratio_actor_local_path = os.path.join(global_step_folder, 'ratio_actor.pt')
        if os.path.exists(ratio_actor_local_path):
            ratio_actor_state_dict = torch.load(ratio_actor_local_path, weights_only=False)
            ray.get(self.ratio_actor.set_state.remote(ratio_actor_state_dict))
            state_dict = ray.get(self.ratio_actor.get_state.remote())
            self.train_dataset.dataframe.sync_with_all_datasets({**state_dict, 'global_step': self.global_steps})
            if self.config.data.get('sampler', None) is not None:
                # update the sampler with the new attempted ratio list
                self.train_dataloader.sampler.attempted_ratio_list = state_dict['attempted_ratio_list']

        else:
            print(f"Warning: Checkpoint {ratio_actor_local_path} does not exist. "
                  f"Using the default ratio actor state dict.")

        
    def log(self, data, reward_extra_info):
        portion = reward_extra_info['portion']
        mean_portion = np.mean(portion)
        std_portion = np.std(portion)
        is_train_set =  data.non_tensor_batch['extra_info'][0]['split']
        stage = 'train' if is_train_set=='train' else 'val' 
        wandb.log({f'portions/{stage}_portions_mean': mean_portion}, step=self.global_steps)
        wandb.log({f'portions/{stage}_portions_std': std_portion}, step=self.global_steps)


    def update_datasets_with_ratios(self, data, scores, reward_extra_info):
        ids = reward_extra_info['index']
        portion = reward_extra_info['portion']
        if data.non_tensor_batch['extra_info'][0]['split'] == 'train': # we are in train mode not validation
            # Update the training dataset
            # ray.get(self.ratio_actor.update_attempted_ratios.remote([(ids, portion, scores)]))
            self.ratio_actor.update_attempted_ratios.remote([(ids, portion, scores)])
            self.ratio_actor.set_global_step.remote(self.global_steps)
            ray.get(self.ratio_actor.update_min_max_avg_ratios.remote())
            # self.ratio_actor.update_min_max_avg_ratios.remote()
            state_dict = ray.get(self.ratio_actor.get_state.remote())
            # print(f'max_per_sample_ratio: {state_dict["max_per_sample_ratio"]}')
            # pprint.pp(f'attempted_ratio list \n {state_dict["attempted_ratios_list"]}')
        
            self.train_dataset.dataframe.sync_with_all_datasets({**state_dict, 'global_step': self.global_steps})
            if self.config.data.get('sampler', None) is not None:
                self.train_dataloader.sampler.attempted_ratio_list = state_dict['attempted_ratios_list']


from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import datasets

class AdaptiveRLHFDataset(RLHFDataset):
    """
    This class is used to train a PPO model with a non-parquet dataset.
    """

    def __init__(self, *args, **kwargs):
        self.type = kwargs.pop('type', 'base')
        self.size = kwargs.pop('size', None)
        self.curriculum_config = kwargs.pop('curriculum_config', {})
        print(args, kwargs)
        super().__init__(*args, **kwargs)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            if self.size is not None:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"].select(range(self.size))
            else:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            answer_key = 'answer'
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True) + tokenizer.encode(doc[answer_key], add_special_tokens=False)
                               ) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")
            
            # to test: tokenizer.decode(tokenizer.apply_chat_template(self.dataframe[0][self.prompt_key], add_generation_prompt=True) + tokenizer.encode(self.dataframe[0]['answer']))

            print(f'filter dataset len: {len(self.dataframe)}')
        if self.type == 'base':
            self.dataframe = CurriculumDatasetWrapper(self.dataframe, **self.curriculum_config)
        elif self.type == 'adaptive':
            self.dataframe = PerSampleCurriculumDatasetWrapper(self.dataframe, **self.curriculum_config)
        else:
            raise NotImplementedError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        chat = row_dict.pop(self.prompt_key)

        # add the partial answer to the chat, if provided
        if row_dict['extra_info'].get('partial_answer', None) is not None:
            partial_answer = row_dict['extra_info']['partial_answer']
            chat.append({'role': 'assistant', 'content': partial_answer})
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, continue_final_message=True, tokenize=False)
        else: 
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        
        raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict 
    
