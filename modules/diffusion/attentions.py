import torch
from diffusers.models.attention_processor import (
    Attention,
    SlicedAttnProcessor,
    XFormersAttnProcessor,
)

try:
    import xformers.ops
except:
    xformers = None


def xformers_forward(
    self: XFormersAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    # Apply hypernetwork if present
    if hasattr(attn, "hypernetwork") and attn.hypernetwork is not None:
        context_k, context_v = attn.hypernetwork.forward(
            hidden_states, encoder_hidden_states
        )
        context_k = context_k.to(hidden_states.dtype)
        context_v = context_v.to(hidden_states.dtype)
    else:
        context_k = encoder_hidden_states
        context_v = encoder_hidden_states

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)

    query = attn.head_to_batch_dim(query).contiguous()
    key = attn.head_to_batch_dim(key).contiguous()
    value = attn.head_to_batch_dim(value).contiguous()

    hidden_states = xformers.ops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attention_mask,
        op=self.attention_op,
        scale=attn.scale,
    )
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def sliced_attn_forward(
    self: SlicedAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = attn.to_q(hidden_states)
    dim = query.shape[-1]
    query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    # Apply hypernetwork if present
    if hasattr(attn, "hypernetwork") and attn.hypernetwork is not None:
        context_k, context_v = attn.hypernetwork.forward(
            hidden_states, encoder_hidden_states
        )
        context_k = context_k.to(hidden_states.dtype)
        context_v = context_v.to(hidden_states.dtype)
    else:
        context_k = encoder_hidden_states
        context_v = encoder_hidden_states

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    batch_size_attention, query_tokens, _ = query.shape
    hidden_states = torch.zeros(
        (batch_size_attention, query_tokens, dim // attn.heads),
        device=query.device,
        dtype=query.dtype,
    )

    for i in range(batch_size_attention // self.slice_size):
        start_idx = i * self.slice_size
        end_idx = (i + 1) * self.slice_size

        query_slice = query[start_idx:end_idx]
        key_slice = key[start_idx:end_idx]
        attn_mask_slice = (
            attention_mask[start_idx:end_idx] if attention_mask is not None else None
        )

        attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

        attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

        hidden_states[start_idx:end_idx] = attn_slice

    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def replace_attentions_for_hypernetwork():
    import diffusers.models.attention_processor

    diffusers.models.attention_processor.XFormersAttnProcessor.__call__ = (
        xformers_forward
    )
    diffusers.models.attention_processor.SlicedAttnProcessor.__call__ = (
        sliced_attn_forward
    )
