# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from torch.distributed._tensor import Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoTokenizer,
    GenerationConfig,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


# to use:
# python -m torch.distributed.run --nproc_per_node=8 --master_port=1234 scripts/model_merger.py --hf_model_path <huggingface_model_path> --local_dir <local_dir> --target_dir <target_dir> --hf_upload_path <huggingface_repo_path>
# python scripts/model_merger.py --hf_model_path <huggingface_model_path> --local_dir <local_dir> --target_dir <target_dir> --hf_upload_path <huggingface_repo_path>

# python scripts/model_merger.py \
#   --hf_model_path meta-llama/Llama-3.2-1B \
#   --local_dir llama-1B-base/actor/ \

# python scripts/model_merger.py \
#   --hf_model_path masani/SFT_cumulative_parity_length_16_bitwidth_1_1024_512_Qwen2-1.5B_epoch_25_global_step_100 \
#   --local_dir /dlabscratch1/amani/LLM-RL/logs/progressive_rl_grpo_on_cumulative_parity_length_16_bitwidth_1_1024_512_SFT_cumulative_parity_length_16_bitwidth_1_1024_512_Qwen2-1.5B_epoch_25_global_step_100/2025-05-02_14-53-16/checkpoints/global_step_6000/actor \
#   --hf_upload_path masani/SFT_cumulative_parity_length_16_bitwidth_1_1024_512_Qwen2-1.5B_6000_RL \
#   --target_dir None \
#   --test \
#   --test_hf_dir <test_hf_dir>
parser = argparse.ArgumentParser()
parser.add_argument("--tie-word-embedding", action="store_true", help="Whether to tie word embedding weights")
parser.add_argument("--is-value-model", action="store_true", help="Whether the model loaded as value model")
parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
parser.add_argument(
    "--local_dir",
    type=str,
    required=True,
    help=("The path for your saved model. For megatron, point to the base dir of model, rng, optimizer checkpoints, commonly be `config.default_local_dir/global_step_\{global_step\}`."),
)
parser.add_argument("--target_dir", required=False, default="tmp", type=str, help="The path for the target model")
parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
parser.add_argument("--test", action="store_true", help="test correctness of hf_model")
parser.add_argument(
    "--test_hf_dir",
    type=str,
    required=False,
    help="test correctness of hf_model, , with hf_model in checkpoint.contents",
)
parser.add_argument("--private", required=False, default=False, help="Whether to upload the model to private repo")

args = parser.parse_args()
os.makedirs(args.target_dir, exist_ok=True)
if args.test:
    assert args.test_hf_dir is not None, "You must run verl save checkpoint first, with hf_model in checkpoint.contents, and provide the directory here"


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(hf_path):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.hf_upload_path, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=hf_path, repo_id=args.hf_upload_path, repo_type="model")


def test_fsdp_state_dict(
    auto_model_class,
    original_hf_model_path: str,
    collected_state_dict: Dict[str, torch.Tensor],
) -> bool:
    # load original model using bf16 since we collected state_dict with bf16
    original_model = auto_model_class.from_pretrained(original_hf_model_path, torch_dtype=torch.bfloat16)
    original_state_dict = original_model.state_dict()
    del original_model  # Free memory

    original_keys = set(original_state_dict.keys())
    collected_keys = set(collected_state_dict.keys())

    missing_keys = original_keys - collected_keys
    assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

    extra_keys = collected_keys - original_keys
    assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

    for key in original_keys:
        original_shape = original_state_dict[key].shape
        collected_shape = collected_state_dict[key].shape
        assert original_shape == collected_shape, f"Shape mismatch for key '{key}': original {original_shape} vs collected {collected_shape}"

        original_dtype = original_state_dict[key].dtype
        collected_dtype = collected_state_dict[key].dtype
        assert original_dtype == collected_dtype, f"Dtype mismatch for key '{key}': original {original_dtype} vs collected {collected_dtype}"

        torch.testing.assert_close(original_state_dict[key], collected_state_dict[key], atol=1e-4, rtol=1e-4)

    print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")
    return True


def patch_model_generation_config(model, hf_model_path):
    """
    The generation_config created from model config may be different to the pretrained model,
    this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

    This function patch the generation_config created from model config to the pretrained model.
    """
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(hf_model_path)
        except OSError:
            print(f"Warning: Generation config file not found in {hf_model_path}, using a generation config created from the model config.")
            pass
    return model


def convert_fsdp_checkpoints_to_hfmodels():
    local_dir = args.local_dir

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break
    assert world_size, "No model file with the proper format"

    state_dict = torch.load(os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt"), map_location="cpu", weights_only=False)
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]

    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing model shards with {total_shards} {mesh_shape} in total")

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print("-" * 30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == "dp" or mesh_dim_names[0] == "ddp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        if key in param_placements:
            # merge shards
            placements: Tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    hf_path = os.path.join(local_dir, "huggingface") if args.target_dir is None else args.target_dir
    config = AutoConfig.from_pretrained(args.hf_model_path)

    if "ForTokenClassification" in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif "ForCausalLM" in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif "ForConditionalGeneration" in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {config['architectures']}")

    if args.test:
        print("Running compatibility test")
        test_fsdp_state_dict(auto_model, args.test_hf_dir, state_dict)

    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model = patch_model_generation_config(model, args.hf_model_path)

    print(f"Saving model to {hf_path}")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model

    print("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    tokenizer.save_pretrained(hf_path)

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path)


if __name__ == "__main__":
    convert_fsdp_checkpoints_to_hfmodels()
