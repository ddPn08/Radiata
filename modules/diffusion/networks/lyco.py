import os

import safetensors.torch
import torch
from lycoris import kohya
from lycoris.modules import locon, loha

kohya.LycorisNetwork.UNET_TARGET_REPLACE_MODULE.remove("Attention")


class LoConModule(locon.LoConModule):
    def make_weight(self):
        org_weight = self.org_module[0].weight.to(torch.float)
        up = self.lora_up.weight.to(device=org_weight.device, dtype=org_weight.dtype)
        down = self.lora_down.weight.to(
            device=org_weight.device, dtype=org_weight.dtype
        )
        if self.cp:
            mid = self.lora_mid.weight.to(
                device=org_weight.device, dtype=org_weight.dtype
            )
            up = up.reshape(up.size(0), up.size(1))
            down = down.reshape(down.size(0), down.size(1))
            weight = torch.einsum("m n w h, i m, n j -> i j w h", mid, up, down)
        else:
            weight = up.reshape(up.size(0), -1) @ down.reshape(down.size(0), -1)

        return weight.reshape(org_weight.shape) * self.scale

    def merge_to(self):
        org_weight = self.org_module[0].weight
        weight = self.make_weight() * self.multiplier
        org_weight.copy_(org_weight + weight.to(org_weight.dtype))


class LohaModule(loha.LohaModule):
    def make_weight(self):
        org_weight = self.org_module[0].weight.to(torch.float)
        w1a = self.hada_w1_a.to(device=org_weight.device, dtype=org_weight.dtype)
        w1b = self.hada_w1_b.to(device=org_weight.device, dtype=org_weight.dtype)
        w2a = self.hada_w2_a.to(device=org_weight.device, dtype=org_weight.dtype)
        w2b = self.hada_w2_b.to(device=org_weight.device, dtype=org_weight.dtype)

        if self.cp:
            t1 = self.hada_t1.to(device=org_weight.device, dtype=org_weight.dtype)
            t2 = self.hada_t2.to(device=org_weight.device, dtype=org_weight.dtype)
            weight_1 = torch.einsum("i j k l, j r -> i r k l", t1, w1b)
            weight_1 = torch.einsum("i j k l, i r -> r j k l", weight_1, w1a)
            weight_2 = torch.einsum("i j k l, j r -> i r k l", t2, w2b)
            weight_2 = torch.einsum("i j k l, i r -> r j k l", weight_2, w2a)
        else:
            weight_1 = w1a @ w1b
            weight_2 = w2a @ w2b
        return (weight_1 * weight_2).reshape(org_weight.shape) * self.scale

    def merge_to(self):
        org_weight = self.org_module[0].weight
        weight = self.make_weight() * self.multiplier
        org_weight.copy_(org_weight + weight.to(org_weight.dtype))


def get_metadata(algo: str, weight):
    if algo == "lora":
        use_cp = False
        conv_alpha = None
        conv_lora_dim = None
        lora_alpha = None
        lora_dim = None
        for key, value in weight.items():
            if key.endswith("alpha"):
                base_key = key[:-6]

                def get_dim():
                    lora_up = weight[f"{base_key}.lora_up.weight"].size()[1]
                    lora_down = weight[f"{base_key}.lora_down.weight"].size()[0]
                    assert (
                        lora_up == lora_down
                    ), "lora_up and lora_down must be same size"
                    return lora_up

                if any([x for x in ["conv", "conv1", "conv2"] if base_key.endswith(x)]):
                    conv_alpha = int(value)
                    conv_lora_dim = get_dim()
                else:
                    lora_alpha = int(value)
                    lora_dim = get_dim()
                if f"{base_key}.lora_mid.weight" in weight:
                    use_cp = True
        return conv_alpha, conv_lora_dim, lora_alpha, lora_dim, {"use_cp": use_cp}
    elif algo == "loha":
        use_cp = False
        conv_alpha = None
        conv_lora_dim = None
        lora_alpha = None
        lora_dim = None
        for key, value in weight.items():
            if key.endswith("alpha"):
                base_key = key[:-6]

                def get_dim():
                    hada_w1_b = weight[f"{base_key}.hada_w1_b"].size()[0]
                    hada_w2_b = weight[f"{base_key}.hada_w2_b"].size()[0]
                    assert (
                        hada_w1_b == hada_w2_b
                    ), "hada_w1_b and hada_w2_b must be same size"
                    return hada_w1_b

                if any([x for x in ["conv", "conv1", "conv2"] if base_key.endswith(x)]):
                    conv_alpha = int(value)
                    conv_lora_dim = get_dim()
                else:
                    lora_alpha = int(value)
                    lora_dim = get_dim()
                if f"{base_key}.hada_t1" in weight and f"{base_key}.hada_t2" in weight:
                    use_cp = True
        return conv_alpha, conv_lora_dim, lora_alpha, lora_dim, {"use_cp": use_cp}


def create_network_from_weights(
    multiplier: float,
    file: str,
    vae,
    text_encoder,
    unet,
    weights_sd: torch.Tensor = None,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            weights_sd = safetensors.torch.load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    algo = None
    apply_unet = None
    apply_te = None
    for key in weights_sd.keys():
        if key.startswith("lora_unet"):
            apply_unet = True
        elif key.startswith("lora_te"):
            apply_te = True
        if "lora_up" in key or "lora_down" in key:
            algo = "lora"
        elif "hada" in key:
            algo = "loha"

        if apply_unet is not None and apply_te is not None and algo is not None:
            break

    if algo is None:
        raise ValueError("Could not determine network module")
    (
        conv_alpha,
        conv_dim,
        lora_alpha,
        lora_dim,
        additional_kwargs,
    ) = get_metadata(algo, weights_sd)
    if lora_dim is None or lora_alpha is None:
        lora_dim = 0
        lora_alpha = 0
    if conv_dim is None or conv_alpha is None:
        conv_dim = 0
        conv_alpha = 0

    network_module = {
        "lora": LoConModule,
        "locon": LoConModule,
        "loha": LohaModule,
        # "ia3": IA3Module,
        # "lokr": LokrModule,
        # "dylora": DyLoraModule,
        # "glora": GLoRAModule,
    }[algo]
    network = LycorisNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=lora_dim,
        conv_lora_dim=conv_dim,
        alpha=lora_alpha,
        conv_alpha=conv_alpha,
        network_module=network_module,
        apply_unet=apply_unet,
        apply_te=apply_te,
        **additional_kwargs,
    )
    return network, weights_sd


class LycorisNetwork(kohya.LycorisNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.apply_unet = kwargs.get("apply_unet", True)
        self.apply_te = kwargs.get("apply_te", True)

        if self.apply_unet:
            for lora in self.unet_loras:
                self.add_module(lora.lora_name, lora)

        if self.apply_te:
            for lora in self.text_encoder_loras:
                self.add_module(lora.lora_name, lora)

        for lora in self.text_encoder_loras + self.unet_loras:
            org_module = lora.org_module[0]
            if not hasattr(org_module, "_lora_org_forward"):
                setattr(org_module, "_lora_org_forward", org_module.forward)
            if not hasattr(org_module, "_lora_org_weight"):
                setattr(org_module, "_lora_org_weight", org_module.weight.clone().cpu())

    def apply_to(self):
        return super().apply_to(None, None, self.apply_te, self.apply_unet)

    def merge_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.merge_to()

    def restore(self, *args):
        for lora in self.text_encoder_loras + self.unet_loras:
            org_module = lora.org_module[0]
            if hasattr(org_module, "_lora_org_forward"):
                org_module.forward = org_module._lora_org_forward
                del org_module._lora_org_forward
            if hasattr(org_module, "_lora_org_weight"):
                org_module.weight.copy_(org_module._lora_org_weight)
                del org_module._lora_org_weight
