import math
import os
from typing import *

import torch
from safetensors.torch import load_file


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
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha = modules_dim[key]

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
    )
    return network, weights_sd


class LoRAModule(torch.nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self,
        lora_name: str,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        lora_alpha: int = 1,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.org_forward = org_module.forward
        self.org_module = [org_module]

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
            )
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(lora_alpha) == torch.Tensor:
            lora_alpha = lora_alpha.detach().cpu().numpy()

        lora_alpha = (
            self.lora_dim if lora_alpha is None or lora_alpha == 0 else lora_alpha
        )
        self.scale = lora_alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(lora_alpha))

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def apply_to(self):
        self.org_module[0].forward = self.forward

    def make_weight(self):
        org_weight = self.org_module[0].weight.to(torch.float)

        state_dict = self.state_dict()
        up_weight = state_dict["lora_up.weight"].to(torch.float).to(org_weight.device)
        down_weight = (
            state_dict["lora_down.weight"].to(torch.float).to(org_weight.device)
        )

        if len(org_weight.size()) == 2:
            # linear
            org_weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            org_weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            org_weight = self.multiplier * conved * self.scale
        return org_weight

    def merge_to(self):
        # extract weight from org_module
        org_weight = self.org_module[0].weight
        weight = self.make_weight()
        # set weight to org_module
        org_weight.copy_(org_weight + weight.to(org_weight.dtype))

    def forward(self, x: torch.Tensor):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(torch.nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRANetwork
    """

    NUM_OF_BLOCKS = 12

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier: float = 1.0,
        modules_dim: Optional[Dict] = None,
        modules_alpha: Optional[Dict] = None,
        network_module: Optional[torch.nn.Module] = LoRAModule,
    ):
        super().__init__()
        self.multiplier = multiplier

        def create_modules(
            is_unet, root_module: torch.nn.Module, target_replace_modules
        ) -> List[LoRAModule]:
            if root_module is None:
                return []
            prefix = (
                LoRANetwork.LORA_PREFIX_UNET
                if is_unet
                else LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            )
            loras = []
            replaced = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]

                            if dim is None or dim == 0:
                                continue

                            lora = network_module(
                                lora_name, child_module, self.multiplier, dim, alpha
                            )
                            loras.append(lora)
                            replaced.append(child_module)
            return loras, replaced

        text_encoder_loras, te_replaced_modules = create_modules(
            False,
            text_encoder,
            LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
        )
        unet_loras, unet_replaced_modules = create_modules(
            True,
            unet,
            LoRANetwork.UNET_TARGET_REPLACE_MODULE
            + LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
        )

        self.text_encoder_loras: List[LoRAModule] = text_encoder_loras
        self.unet_loras: List[LoRAModule] = unet_loras

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
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
            if not hasattr(module, "_lora_org_weight"):
                setattr(module, "_lora_org_weight", module.weight.clone().cpu())

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def apply_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()

    def merge_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.merge_to()

    def restore(self, *modules: torch.nn.Module):
        for module in modules:
            for child in module.modules():
                if hasattr(child, "_lora_org_forward"):
                    child.forward = child._lora_org_forward
                    del child._lora_org_forward
                if hasattr(child, "_lora_org_weight"):
                    child.weight.copy_(child._lora_org_weight)
                    del child._lora_org_weight
