#!fork: https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/kohya.py
import os
from typing import *

import torch
from safetensors.torch import load_file

from .lycoris.locon import LoConModule
from .lycoris.loha import LohaModule


def create_network_from_weights(
    multiplier: float,
    file: str,
    text_encoder,
    unet,
    weights_sd: torch.Tensor = None,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    network_module = None
    lora_dim = None
    lora_alpha = None
    conv_lora_dim = None
    conv_alpha = None
    additional_kwargs = {}
    for key, value in weights_sd.items():
        if network_module is None:
            if "lora_up" in key or "lora_down" in key:
                network_module = LoConModule
                break
            elif "hada" in key:
                network_module = LohaModule
                break

    (
        conv_alpha,
        conv_lora_dim,
        lora_alpha,
        lora_dim,
        additional_kwargs,
    ) = network_module.get_metadata(weights_sd)

    if network_module is None:
        raise ValueError("Could not determine network module")
    if lora_dim is None or lora_alpha is None:
        lora_dim = 0
        lora_alpha = 0
    if conv_lora_dim is None or conv_alpha is None:
        conv_lora_dim = 0
        conv_alpha = 0

    network = LycorisNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        network_module=network_module,
        lora_dim=lora_dim,
        alpha=lora_alpha,
        conv_lora_dim=conv_lora_dim,
        conv_alpha=conv_alpha,
        **additional_kwargs,
    )
    return network, weights_sd


class LycorisNetwork(torch.nn.Module):
    # Ignore proj_in or proj_out, their channels is only a few.
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        conv_lora_dim=4,
        alpha=1,
        conv_alpha=1,
        use_cp=False,
        dropout=0,
        network_module=LoConModule,
        **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.conv_lora_dim = int(conv_lora_dim)
        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        self.dropout = dropout

        # create module instances
        def create_modules(
            prefix,
            root_module: torch.nn.Module,
            target_replace_modules,
            target_replace_names=[],
        ) -> List[network_module]:
            loras = []
            replaced_module = []
            if root_module is None:
                return loras
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        if child_module.__class__.__name__ == "Linear" and lora_dim > 0:
                            lora = network_module(
                                lora_name,
                                child_module,
                                self.multiplier,
                                self.lora_dim,
                                self.alpha,
                                self.dropout,
                                use_cp,
                                **kwargs,
                            )
                        elif child_module.__class__.__name__ == "Conv2d":
                            k_size, *_ = child_module.kernel_size
                            if k_size == 1 and lora_dim > 0:
                                lora = network_module(
                                    lora_name,
                                    child_module,
                                    self.multiplier,
                                    self.lora_dim,
                                    self.alpha,
                                    self.dropout,
                                    use_cp,
                                    **kwargs,
                                )
                            elif conv_lora_dim > 0:
                                lora = network_module(
                                    lora_name,
                                    child_module,
                                    self.multiplier,
                                    self.conv_lora_dim,
                                    self.conv_alpha,
                                    self.dropout,
                                    use_cp,
                                    **kwargs,
                                )
                            else:
                                continue
                        else:
                            continue
                        loras.append(lora)
                        replaced_module.append(child_module)
                elif name in target_replace_names:
                    lora_name = prefix + "." + name
                    lora_name = lora_name.replace(".", "_")
                    if module.__class__.__name__ == "Linear" and lora_dim > 0:
                        lora = network_module(
                            lora_name,
                            module,
                            self.multiplier,
                            self.lora_dim,
                            self.alpha,
                            self.dropout,
                            use_cp,
                            **kwargs,
                        )
                    elif module.__class__.__name__ == "Conv2d":
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            lora = network_module(
                                lora_name,
                                module,
                                self.multiplier,
                                self.lora_dim,
                                self.alpha,
                                self.dropout,
                                use_cp,
                                **kwargs,
                            )
                        elif conv_lora_dim > 0:
                            lora = network_module(
                                lora_name,
                                module,
                                self.multiplier,
                                self.conv_lora_dim,
                                self.conv_alpha,
                                self.dropout,
                                use_cp,
                                **kwargs,
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)
                    replaced_module.append(module)
            return loras, replaced_module

        self.text_encoder_loras, te_replaced_modules = create_modules(
            LycorisNetwork.LORA_PREFIX_TEXT_ENCODER,
            text_encoder,
            LycorisNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
        )

        self.unet_loras, unet_replaced_modules = create_modules(
            LycorisNetwork.LORA_PREFIX_UNET,
            unet,
            LycorisNetwork.UNET_TARGET_REPLACE_MODULE,
        )

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
            self.add_module(lora.lora_name, lora)

        for module in te_replaced_modules + unet_replaced_modules:
            if not hasattr(module, "_lora_org_forward"):
                setattr(module, "_lora_org_forward", module.forward)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def apply_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()

    def restore(self, *modules: torch.nn.Module):
        for module in modules:
            for child in module.modules():
                if hasattr(child, "_lora_org_forward"):
                    child.forward = child._lora_org_forward
                    del child._lora_org_forward
