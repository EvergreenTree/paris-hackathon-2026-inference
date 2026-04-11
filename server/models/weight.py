from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from server.distributed import get_tp_info
from server.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _slice_heads(
    value: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rank: int,
    world_size: int,
    dim: int = 0,
    allow_replicate: bool = False,
) -> torch.Tensor:
    if allow_replicate and num_heads < world_size:
        head_idx = rank * num_heads // world_size
        start = head_idx * head_dim
        return value.narrow(dim, start, head_dim).clone()
    return value.chunk(world_size, dim=dim)[rank].clone()


def _shard_qwen35_linear_attn_tensor(key: str, value: torch.Tensor, r: int, n: int, config):
    if ".linear_attn." not in key:
        return None

    key_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    key_dim = config.linear_key_head_dim
    value_dim = config.linear_value_head_dim
    total_key_dim = key_heads * key_dim
    total_value_dim = value_heads * value_dim

    if key.endswith(".in_proj_qkv.weight") or key.endswith(".conv1d.weight"):
        q, k, v = value.split([total_key_dim, total_key_dim, total_value_dim], dim=0)
        return torch.cat(
            [
                _slice_heads(
                    q,
                    num_heads=key_heads,
                    head_dim=key_dim,
                    rank=r,
                    world_size=n,
                    allow_replicate=True,
                ),
                _slice_heads(
                    k,
                    num_heads=key_heads,
                    head_dim=key_dim,
                    rank=r,
                    world_size=n,
                    allow_replicate=True,
                ),
                _slice_heads(
                    v,
                    num_heads=value_heads,
                    head_dim=value_dim,
                    rank=r,
                    world_size=n,
                ),
            ],
            dim=0,
        )

    if key.endswith(".in_proj_z.weight"):
        return _slice_heads(
            value,
            num_heads=value_heads,
            head_dim=value_dim,
            rank=r,
            world_size=n,
        )

    if key.endswith((".in_proj_b.weight", ".in_proj_a.weight", ".dt_bias", ".A_log")):
        return _slice_heads(
            value,
            num_heads=value_heads,
            head_dim=1,
            rank=r,
            world_size=n,
        )

    return None


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, config):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if config.is_qwen3_5_moe:
        sharded = _shard_qwen35_linear_attn_tensor(key, value, r, n, config)
        if sharded is not None:
            return sharded
        if ".experts.gate_up_proj" in key:
            return value.chunk(n, dim=1)[r].clone()
        if ".experts.down_proj" in key:
            return value.chunk(n, dim=2)[r].clone()

    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and config.num_kv_heads is not None and config.num_kv_heads < n:
            head_dim = value.shape[0] // config.num_kv_heads
            head_idx = r * config.num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefixes and skip vision/projector weights.
                if name.startswith(
                    (
                        "vision_tower.",
                        "multi_modal_projector.",
                        "model.visual.",
                        "visual.",
                    )
                ):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                if name.startswith("model.language_model."):
                    name = "model." + name.removeprefix("model.language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
