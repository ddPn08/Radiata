#!fork: https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/locon.py
import math

import torch
import torch.nn as nn


def get_dim(weight, base_key: str):
    lora_up = weight[f"{base_key}.lora_up.weight"].size()[1]
    lora_down = weight[f"{base_key}.lora_down.weight"].size()[0]
    assert lora_up == lora_down, "lora_up and lora_down must be same size"
    return lora_up


class LoConModule(nn.Module):
    @classmethod
    def get_metadata(cls, weight):
        use_cp = False
        for key, value in weight.items():
            if key.endswith("alpha"):
                base_key = key[:-6]
                if any([x for x in ["conv", "conv1", "conv2"] if base_key.endswith(x)]):
                    conv_alpha = int(value)
                    conv_lora_dim = get_dim(weight, base_key)
                else:
                    lora_alpha = int(value)
                    lora_dim = get_dim(weight, base_key)
                if f"{base_key}.lora_mid.weight" in weight:
                    use_cp = True
        return conv_alpha, conv_lora_dim, lora_alpha, lora_dim, {"use_cp": use_cp}

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        use_cp=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False
        self.org_forward = org_module.forward
        self.multiplier = multiplier
        self.org_module = [org_module]

        if org_module.__class__.__name__ == "Conv2d":
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            if use_cp and k_size != (1, 1):
                self.lora_down = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
                self.lora_mid = nn.Conv2d(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.cp = True
            else:
                self.lora_down = nn.Conv2d(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        self.shape = org_module.weight.shape

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        if self.cp:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

    def apply_to(self):
        self.org_module[0].forward = self.forward

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
        weight = self.make_weight()
        org_weight.copy_(org_weight + weight.to(org_weight.dtype))

    def forward(self, x):
        if self.cp:
            return self.org_forward(x) + self.dropout(
                self.lora_up(self.lora_mid(self.lora_down(x)))
                * self.multiplier
                * self.scale
            )
        else:
            return self.org_forward(x) + self.dropout(
                self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
            )
