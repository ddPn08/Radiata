# Fake Ancestral sampler for multidiffusion
# (random noise was removed and sigma_up was extracted)
from dataclasses import dataclass
from typing import *

import torch
from diffusers import EulerAncestralDiscreteScheduler, KDPM2AncestralDiscreteScheduler
from torch import FloatTensor


@dataclass
class SamplerOutput:
    prev_sample: torch.FloatTensor
    sigma_up: torch.FloatTensor


class EulerAncestralSampler(EulerAncestralDiscreteScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> SamplerOutput | Tuple:
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = (
            sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
        ) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        if not return_dict:
            return (prev_sample, sigma_up)

        return SamplerOutput(prev_sample=prev_sample, sigma_up=sigma_up)


class KDPM2AncestralSampler(KDPM2AncestralDiscreteScheduler):
    def step(
        self,
        model_output: FloatTensor,
        timestep: float | FloatTensor,
        sample: FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> SamplerOutput:
        step_index = self.index_for_timestep(timestep)

        if self.state_in_first_order:
            sigma = self.sigmas[step_index]
            sigma_interpol = self.sigmas_interpol[step_index]
            sigma_up = self.sigmas_up[step_index]
            sigma_down = self.sigmas_down[step_index - 1]
        else:
            # 2nd order / KPDM2's method
            sigma = self.sigmas[step_index - 1]
            sigma_interpol = self.sigmas_interpol[step_index - 1]
            sigma_up = self.sigmas_up[step_index - 1]
            sigma_down = self.sigmas_down[step_index - 1]

        # currently only gamma=0 is supported. This usually works best anyways.
        # We can support gamma in the future but then need to scale the timestep before
        # passing it to the model which requires a change in API
        gamma = 0
        sigma_hat = sigma * (gamma + 1)  # Note: sigma_hat == sigma for now

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            sigma_input = sigma_hat if self.state_in_first_order else sigma_interpol
            pred_original_sample = sample - sigma_input * model_output
        elif self.config.prediction_type == "v_prediction":
            sigma_input = sigma_hat if self.state_in_first_order else sigma_interpol
            pred_original_sample = model_output * (
                -sigma_input / (sigma_input**2 + 1) ** 0.5
            ) + (sample / (sigma_input**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        if self.state_in_first_order:
            # 2. Convert to an ODE derivative for 1st order
            derivative = (sample - pred_original_sample) / sigma_hat
            # 3. delta timestep
            dt = sigma_interpol - sigma_hat

            # store for 2nd order step
            self.sample = sample
            self.dt = dt
            prev_sample = sample + derivative * dt
            sigma_up = None
        else:
            # DPM-Solver-2
            # 2. Convert to an ODE derivative for 2nd order
            derivative = (sample - pred_original_sample) / sigma_interpol
            # 3. delta timestep
            dt = sigma_down - sigma_hat

            sample = self.sample
            self.sample = None

            prev_sample = sample + derivative * dt

        if not return_dict:
            return (prev_sample, sigma_up)

        return SamplerOutput(prev_sample=prev_sample, sigma_up=sigma_up)
