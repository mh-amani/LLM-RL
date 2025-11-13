import yaml

# def get_model_name(model):
#    with open(f'config/model/{model}.yaml', 'r') as f:
#       data = yaml.safe_load(f)
#       return data['model_name_or_path'], data['model_name']

# counter = 0
# base_counter = 20
# for model in ['gpt2xl_1.5b', 'phi4_4b', 'qwen_0.5b', 'qwen_1.5b', 'qwen_7b', 'qwen_instr_0.5b', 'qwen_instr_1.5b', 'qwen_instr_7b']:
#   for gt in [0.125, 0.250, 0.5, 0.625, 0.875]:
#     for dataset in ['gsm8k', 'math']:
#         completion_len = "" if dataset == 'gsm8k' else "trainer_args.max_completion_length=1000"
#         template = f'''- id: {base_counter + counter}
#   name: "{get_model_name(model)[1]}-{dataset}-adaptive-{gt}-id-{base_counter + counter}"
#   model: "{get_model_name(model)[0]}"
#   command: "accelerate launch --num_processes 7 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=adaptive trainer_args=vllm dataset_wrapper.reward_threshold={gt} model={model} task={dataset} {completion_len}"'''
#         counter += 1
#         print(template)


# counter = 0
# base_counter = 100
# for model in ['gpt2xl_1.5b', 'phi4_4b', 'qwen_0.5b', 'qwen_1.5b', 'qwen_7b', 'qwen_instr_0.5b', 'qwen_instr_1.5b', 'qwen_instr_7b']:
#     for dataset in ['gsm8k', 'math']:
#         completion_len = "" if dataset == 'gsm8k' else "trainer_args.max_completion_length=1000"
#         template = f'''- id: {base_counter + counter}
#   name: "{get_model_name(model)[1]}-{dataset}-baseline-id-{base_counter + counter}"
#   model: "{get_model_name(model)[0]}"
#   command: "accelerate launch --num_processes 7 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=default trainer_args=vllm model={model} task={dataset} {completion_len}"'''
#         counter += 1
#         print(template)

# print(" ".join(map(str, range(20, 100))))


# Verl - 22 April 2025

counter = 0
base_counter = 611
# 'Qwen/Qwen2.5-1.5B', 'microsoft/rho-math-1b-v0.1', 'Qwen/Qwen2.5-7B', 'aryolotfi/SFT_gsm8k_rho-math-1b-v0.1_epoch_1_global_step_29', 'aryolotfi/SFT_gsm8k_Mistral-7B-v0.1_epoch_2_global_step_58', 'aryolotfi/SFT_gsm8k_Mistral-7B-v0.1_epoch_2_global_step_58', 'masani/SFT_gsm8k_Llama-2-7b-hf_epoch_1_global_step_29', 'aryolotfi/SFT_math_Mistral-7B-v0.1_epoch_2_global_step_58', 'masani/SFT_math_Llama-2-7b-hf_epoch_1_global_step_29'
for model in ['masani/SFT_math_Llama-3.1-8B_epoch_3_global_step_87']:
    for dataset in ['math']:
        more_info = "" if dataset == 'gsm8k' else "actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 train_batch_size=256"
        for algo in ['grpo']:
            for zero_prob in [0.15]:
                template = f'''- id: {base_counter + counter}
  name: "{model}-adaptive-{zero_prob}-id-{base_counter + counter}"
  model: "{model}"
  command: 'python src/train.py experiment={algo} data={dataset} data.train_dataset_type=adaptive model_path={model} data.curriculum_config.zero_prob={zero_prob} {more_info}' '''
                counter += 1
                print(template)
            
            ## baseline run
            template = f'''- id: {base_counter + counter}
  name: "{model}-base-id-{base_counter + counter}"
  model: "{model}"
  command: 'python src/train.py experiment={algo} data={dataset} data.train_dataset_type=base model_path={model} {more_info}' '''
            counter += 1
            print(template)

print("here's the ids:")
print(' '.join(list(map(str, range(base_counter, base_counter + counter)))))