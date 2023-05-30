from losalina import hypernetwork

loaded_networks = []


def apply_single_hypernetwork(
    hypernetwork: hypernetwork, hidden_states, encoder_hidden_states
):
    context_k, context_v = hypernetwork.forward(hidden_states, encoder_hidden_states)
    return context_k, context_v


def apply_hypernetworks(context_k, context_v, layer=None):
    if len(loaded_networks) == 0:
        return context_v, context_v
    for hypernetwork in loaded_networks:
        context_k, context_v = hypernetwork.forward(context_k, context_v)

    context_k = context_k.to(dtype=context_k.dtype)
    context_v = context_v.to(dtype=context_k.dtype)

    return context_k, context_v


def apply_to(self: hypernetwork):
    loaded_networks.append(self)


def restore(self: hypernetwork, *args):
    loaded_networks.clear()


def hijack_hypernetwork():
    hypernetwork.Hypernetwork.apply_to = apply_to
    hypernetwork.Hypernetwork.restore = restore

    import diffusers.models.attention_processor

    from . import attentions

    # replace the forward function of the attention processors for the hypernetworks
    diffusers.models.attention_processor.XFormersAttnProcessor.__call__ = (
        attentions.xformers_forward
    )
    diffusers.models.attention_processor.SlicedAttnProcessor.__call__ = (
        attentions.sliced_attn_forward
    )
    diffusers.models.attention_processor.AttnProcessor2_0.__call__ = (
        attentions.v2_0_forward
    )
