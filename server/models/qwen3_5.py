from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from server.core import get_global_ctx
from server.distributed import DistributedCommunicator, get_tp_info
from server.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    silu_and_mul,
)
from server.layers.rotary import get_rope
from server.utils import div_even, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig, RotaryConfig


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    chunk_size = 64
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(causal_mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    query = query * (query.shape[-1] ** -0.5)
    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype
    )
    last_recurrent_state = initial_state.to(value)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen35RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, *, centered: bool = True) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.centered = centered
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def _runtime_weight(self) -> torch.Tensor:
        return self.weight + 1.0 if self.centered else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self._runtime_weight(), self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self._runtime_weight(), self.eps, out=x)


class Qwen35RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def _runtime_weight(self) -> torch.Tensor:
        return self.weight + 1.0

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self._runtime_weight()
        if residual is None:
            return self.rmsnorm(x, weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, weight, self.eps)
        return x, residual


class Qwen35RMSNormGated(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * self.weight.float() * F.silu(gate.float())
        return out.to(x.dtype)


class LinearLocal(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False) -> None:
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class DepthwiseConv1D(BaseOP):
    def __init__(self, channels: int, kernel_size: int) -> None:
        self.weight = torch.empty(channels, 1, kernel_size)


@dataclass
class _DeltaLayerState:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    valid: set[int] = field(default_factory=set)


class DeltaStateCache:
    def __init__(self, num_slots: int, device: torch.device, dtype: torch.dtype) -> None:
        self.num_slots = num_slots
        self.device = device
        self.dtype = dtype
        self.layers: dict[int, _DeltaLayerState] = {}

    def get_layer(
        self,
        layer_id: int,
        *,
        conv_dim: int,
        conv_kernel_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> _DeltaLayerState:
        if layer_id not in self.layers:
            self.layers[layer_id] = _DeltaLayerState(
                conv_state=torch.zeros(
                    self.num_slots,
                    conv_dim,
                    conv_kernel_size,
                    device=self.device,
                    dtype=self.dtype,
                ),
                recurrent_state=torch.zeros(
                    self.num_slots,
                    num_heads,
                    key_dim,
                    value_dim,
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
        return self.layers[layer_id]

    def free(self, table_idx: int) -> None:
        for state in self.layers.values():
            state.valid.discard(table_idx)


class Qwen35LinearQKVMerged(LinearLocal):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ) -> None:
        tp_info = get_tp_info()
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        local_size = (2 * local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(hidden_size, local_size, has_bias=has_bias)


class Qwen35AttentionLayer(BaseOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: Qwen35RMSNorm,
        k_norm: Qwen35RMSNorm,
    ) -> None:
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q_gate, k, v = qkv.split([2 * self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        q, gate = q_gate.split([self.qo_attn_dim, self.qo_attn_dim], dim=-1)
        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim) * torch.sigmoid(gate)


class Qwen35RopeGatedAttn(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        head_dim = config.head_dim
        self.qkv_proj = Qwen35LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=config.attention_bias,
        )
        self.q_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn = Qwen35AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=config.attention_bias,
        )

    @nvtx_annotate("Qwen35FullAttention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


class Qwen35GatedDeltaNet(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        tp_info = get_tp_info()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_v_heads = div_even(config.linear_num_value_heads, tp_info.size)
        self.num_k_heads = div_even(config.linear_num_key_heads, tp_info.size, allow_replicate=True)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.full_value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv1d = DepthwiseConv1D(self.conv_dim, self.conv_kernel_size)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.A_log = torch.empty(self.num_v_heads)
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearRowParallel(self.full_value_dim, self.hidden_size, has_bias=False)
        self.in_proj_qkv = LinearLocal(self.hidden_size, self.conv_dim, has_bias=False)
        self.in_proj_z = LinearLocal(self.hidden_size, self.value_dim, has_bias=False)
        self.in_proj_b = LinearLocal(self.hidden_size, self.num_v_heads, has_bias=False)
        self.in_proj_a = LinearLocal(self.hidden_size, self.num_v_heads, has_bias=False)

    def _layer_state(self) -> _DeltaLayerState:
        ctx = get_global_ctx()
        cache = ctx.delta_state_cache
        if (
            cache is None
            or cache.device != ctx.page_table.device
            or cache.dtype != self.conv1d.weight.dtype
            or cache.num_slots != ctx.page_table.shape[0]
        ):
            cache = DeltaStateCache(ctx.page_table.shape[0], ctx.page_table.device, self.conv1d.weight.dtype)
            ctx.delta_state_cache = cache
        return cache.get_layer(
            self.layer_id,
            conv_dim=self.conv_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_heads=self.num_v_heads,
            key_dim=self.head_k_dim,
            value_dim=self.head_v_dim,
        )

    def _split_qkv(self, mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

    def _project_delta_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv = self.in_proj_qkv.forward(x)
        z = self.in_proj_z.forward(x)
        b = self.in_proj_b.forward(x)
        a = self.in_proj_a.forward(x)
        return mixed_qkv, z, b, a

    def _finish_delta(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value = self._split_qkv(mixed_qkv)
        batch_size, seq_len = query.shape[0], query.shape[1]
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        core_attn_out, last_state = _recurrent_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size * seq_len, self.value_dim)
        return self.out_proj.forward(core_attn_out), last_state

    def _forward_decode(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        state = self._layer_state()
        table_idxs = batch.table_idxs[: x.shape[0]].to(torch.long)
        mixed_qkv, z, b, a = self._project_delta_inputs(x)

        conv_state = state.conv_state.index_select(0, table_idxs)
        conv_input = torch.cat([conv_state, mixed_qkv.unsqueeze(-1)], dim=-1)
        new_conv_state = conv_input[:, :, -self.conv_kernel_size :]
        conv_out = F.conv1d(conv_input, self.conv1d.weight, groups=self.conv_dim)[:, :, -1]
        state.conv_state.index_copy_(0, table_idxs, new_conv_state)

        mixed_qkv = F.silu(conv_out).unsqueeze(1)
        recurrent_state = state.recurrent_state.index_select(0, table_idxs)
        out, last_state = self._finish_delta(
            mixed_qkv=mixed_qkv,
            z=z.unsqueeze(1),
            b=b.unsqueeze(1),
            a=a.unsqueeze(1),
            recurrent_state=recurrent_state,
        )
        state.recurrent_state.index_copy_(0, table_idxs, last_state)
        return out

    def _forward_prefill(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        state = self._layer_state()
        mixed_qkv, z, b, a = self._project_delta_inputs(x)
        outputs: list[torch.Tensor] = []
        offset = 0
        zero_state = None
        for req in batch.padded_reqs:
            length = req.extend_len
            table_idx = req.table_idx
            mixed = mixed_qkv[offset : offset + length].transpose(0, 1).unsqueeze(0)
            has_state = table_idx in state.valid
            if has_state:
                prev_conv = state.conv_state[table_idx : table_idx + 1]
                conv_input = torch.cat([prev_conv, mixed], dim=-1)
                conv_out = F.conv1d(conv_input, self.conv1d.weight, groups=self.conv_dim)[:, :, -length:]
                next_conv = conv_input[:, :, -self.conv_kernel_size :]
            else:
                conv_input = F.pad(mixed, (self.conv_kernel_size - 1, 0))
                conv_out = F.conv1d(conv_input, self.conv1d.weight, groups=self.conv_dim)
                if length >= self.conv_kernel_size:
                    next_conv = mixed[:, :, -self.conv_kernel_size :]
                else:
                    next_conv = torch.cat(
                        [
                            torch.zeros(
                                1,
                                self.conv_dim,
                                self.conv_kernel_size - length,
                                device=mixed.device,
                                dtype=mixed.dtype,
                            ),
                            mixed,
                        ],
                        dim=-1,
                    )
            state.conv_state[table_idx].copy_(next_conv[0])

            mixed = F.silu(conv_out.transpose(1, 2))
            if has_state:
                recurrent_state = state.recurrent_state[table_idx : table_idx + 1]
                out, last_state = self._finish_delta(
                    mixed_qkv=mixed,
                    z=z[offset : offset + length].unsqueeze(0),
                    b=b[offset : offset + length].unsqueeze(0),
                    a=a[offset : offset + length].unsqueeze(0),
                    recurrent_state=recurrent_state,
                )
            else:
                if zero_state is None or zero_state.shape[1] != self.num_v_heads:
                    zero_state = torch.zeros(
                        1,
                        self.num_v_heads,
                        self.head_k_dim,
                        self.head_v_dim,
                        device=x.device,
                        dtype=torch.float32,
                    )
                query, key, value = self._split_qkv(mixed)
                query = query.reshape(1, length, self.num_k_heads, self.head_k_dim)
                key = key.reshape(1, length, self.num_k_heads, self.head_k_dim)
                value = value.reshape(1, length, self.num_v_heads, self.head_v_dim)
                z_seg = z[offset : offset + length].reshape(1, length, self.num_v_heads, self.head_v_dim)
                beta = b[offset : offset + length].unsqueeze(0).sigmoid()
                a_seg = a[offset : offset + length].unsqueeze(0)
                g = -self.A_log.float().exp() * F.softplus(a_seg.float() + self.dt_bias)
                if self.num_v_heads // self.num_k_heads > 1:
                    repeat = self.num_v_heads // self.num_k_heads
                    query = query.repeat_interleave(repeat, dim=2)
                    key = key.repeat_interleave(repeat, dim=2)
                core_attn_out, last_state = _chunk_gated_delta_rule(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                )
                core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
                core_attn_out = self.norm.forward(core_attn_out, z_seg.reshape(-1, self.head_v_dim))
                out = self.out_proj.forward(core_attn_out.reshape(length, self.value_dim))

            state.recurrent_state[table_idx].copy_(last_state[0])
            state.valid.add(table_idx)
            outputs.append(out)
            offset += length
        return torch.cat(outputs, dim=0)

    @nvtx_annotate("Qwen35GatedDeltaNet")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_decode:
            return self._forward_decode(x)
        return self._forward_prefill(x)


class Qwen35GatedMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(
            intermediate_size,
            hidden_size,
            has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        return self.down_proj.forward(silu_and_mul(gate_up))


class Qwen35MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )
        self.shared_expert = Qwen35GatedMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    @nvtx_annotate("Qwen35MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        shared = self.shared_expert.forward(hidden_states)
        shared = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)) * shared
        return (routed + shared).view(num_tokens, hidden_dim)


class Qwen35DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        layer_types = config.layer_types or ["full_attention"] * config.num_layers
        self.layer_type = layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_id)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35RopeGatedAttn(config, layer_id)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen35MoEMLP(config)
        self.input_layernorm = Qwen35RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Qwen35Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        if self.layer_type == "linear_attention":
            x = self.linear_attn.forward(x)
        else:
            x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen35Model(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen35DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = Qwen35RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self.model = Qwen35Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5MoeForCausalLM", "DeltaStateCache"]
