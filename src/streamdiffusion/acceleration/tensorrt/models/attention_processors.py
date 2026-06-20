from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND


def get_nn_feats_trt(x, y, threshold=0.9):
    """Per-token nearest-neighbour feature merge (StreamV2V feature injection), TRT-traceable.
    x = [B, Sx, C] current tokens; y = [B, Sy, C] banked output tokens. For each x token, find its
    cosine-nearest y token; if the best cosine < threshold keep x, else replace with that y token.
    Mirrors streamv2v utils.get_nn_feats (all ONNX-bakeable ops: normalize/matmul/max/gather/where)."""
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    cos = torch.matmul(x_norm, y_norm.transpose(1, 2))          # [B, Sx, Sy]
    max_cos, nn_idx = torch.max(cos, dim=-1)                    # [B, Sx]
    mask = max_cos < threshold                                 # keep-original where below threshold
    idx = nn_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))      # [B, Sx, C]
    nn = torch.gather(y, 1, idx)                                # gather nearest y tokens
    return torch.where(mask.unsqueeze(-1), x, nn)


class CachedSTAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        kvo_cache: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if kvo_cache is not None:
            cached_key = kvo_cache[0]
            cached_value = kvo_cache[1]
        else:
            cached_key, cached_value = None, None

        if is_selfattn:
            curr_key = key.clone()
            curr_value = value.clone()

            if cached_key is not None:
                cached_key_reshaped = cached_key.transpose(0, 1).contiguous().flatten(1, 2)
                cached_value_reshaped = cached_value.transpose(0, 1).contiguous().flatten(1, 2)
                key = torch.cat([curr_key, cached_key_reshaped], dim=1)
                value = torch.cat([curr_value, cached_value_reshaped], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
            
        if is_selfattn:
            kvo_cache = torch.stack([curr_key.unsqueeze(0), curr_value.unsqueeze(0)], dim=0)

        return hidden_states, kvo_cache


class CachedSTAttnInjectProcessor2_0:
    r"""StreamV2V FULL processor (extended self-attention + feature injection), TRT-bakeable.

    Uses a 3-component per-layer cache [K, V, O] (kvo_cache[0]=keys, [1]=values, [2]=attention OUTPUTS).
    Extended self-attention concatenates banked K/V into attention (as CachedSTAttnProcessor2_0). When
    use_feature_injection (the up_blocks.1 / mid_block self-attn layers), it additionally merges the
    current attention output with its cosine-nearest-neighbour among the banked OUTPUTS (get_nn_feats)
    and blends: out = out*(1-fi) + fi*nn. The banked output is the PRE-injection output (matches the
    original streamv2v processor). ToMe bank compaction is host-side (not baked). All 16 layers emit the
    3-component cache (uniform I/O); only the injection layers consume cached_output."""

    def __init__(self, use_feature_injection=False, feature_injection_strength=0.8,
                 feature_similarity_threshold=0.98):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("requires PyTorch 2.0")
        self.use_feature_injection = use_feature_injection
        self.fi_strength = feature_injection_strength
        self.threshold = feature_similarity_threshold

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, scale: float = 1.0, kvo_cache=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        is_selfattn = encoder_hidden_states is None
        if is_selfattn:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        cached_key = cached_value = cached_output = None
        if kvo_cache is not None:
            cached_key, cached_value, cached_output = kvo_cache[0], kvo_cache[1], kvo_cache[2]

        if is_selfattn:
            curr_key, curr_value = key.clone(), value.clone()
            if cached_key is not None:
                ck = cached_key.transpose(0, 1).contiguous().flatten(1, 2)
                cv = cached_value.transpose(0, 1).contiguous().flatten(1, 2)
                key = torch.cat([curr_key, ck], dim=1)
                value = torch.cat([curr_value, cv], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        kvo_cache_out = None
        if is_selfattn:
            curr_output = hidden_states.clone()  # bank the PRE-injection output (matches original)
            if self.use_feature_injection and cached_output is not None:
                co = cached_output.transpose(0, 1).contiguous().flatten(1, 2)   # [B, maxframes*seq, C]
                nn = get_nn_feats_trt(hidden_states, co, threshold=self.threshold)
                hidden_states = hidden_states * (1.0 - self.fi_strength) + self.fi_strength * nn
            kvo_cache_out = torch.stack(
                [curr_key.unsqueeze(0), curr_value.unsqueeze(0), curr_output.unsqueeze(0)], dim=0)
        return hidden_states, kvo_cache_out