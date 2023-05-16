import diffusers

# TODO: add rest Auto111 samplers
# Commented schedulers are still unavailable in diffusers 0.16.1
SCHEDULERS = {
    "unipc": diffusers.schedulers.UniPCMultistepScheduler,
    "euler_a": diffusers.schedulers.EulerAncestralDiscreteScheduler,
    "euler": diffusers.schedulers.EulerDiscreteScheduler,
    "ddim": diffusers.schedulers.DDIMScheduler,
    "ddpm": diffusers.schedulers.DDPMScheduler,
    "deis": diffusers.schedulers.DEISMultistepScheduler,
    "dpm2": diffusers.schedulers.KDPM2DiscreteScheduler,
    "dpm2-a": diffusers.schedulers.KDPM2AncestralDiscreteScheduler,
    "dpm++_2s": diffusers.schedulers.DPMSolverSinglestepScheduler,
    "dpm++_2m": diffusers.schedulers.DPMSolverMultistepScheduler,
    "dpm++_2m_karras": diffusers.schedulers.DPMSolverMultistepScheduler,
    # "dpm++_sde": diffusers.schedulers.DPMSolverSDEScheduler,
    # "dpm++_sde_karras": diffusers.schedulers.DPMSolverSDEScheduler,
    "heun": diffusers.schedulers.HeunDiscreteScheduler,
    "heun_karras": diffusers.schedulers.HeunDiscreteScheduler,
    "lms": diffusers.schedulers.LMSDiscreteScheduler,
    # "lms_karras": diffusers.schedulers.LMSDiscreteScheduler,
    "pndm": diffusers.schedulers.PNDMScheduler,
}


def parser_schedulers_config(scheduler_id: str):
    """
    add extra config parameter to scheduler
    """
    kwargs = {}
    if "karras" in scheduler_id:
        kwargs["use_karras_sigmas"] = True
    return kwargs
