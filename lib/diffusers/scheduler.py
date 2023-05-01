import diffusers

SCHEDULERS = {
    "unipc": diffusers.schedulers.UniPCMultistepScheduler,
    "euler_a": diffusers.schedulers.EulerAncestralDiscreteScheduler,
    "euler": diffusers.schedulers.EulerDiscreteScheduler,
    "ddim": diffusers.schedulers.DDIMScheduler,
    "ddpm": diffusers.schedulers.DDPMScheduler,
    "deis": diffusers.schedulers.DEISMultistepScheduler,
    "dpm2": diffusers.schedulers.KDPM2DiscreteScheduler,
    "dpm2-a": diffusers.schedulers.KDPM2AncestralDiscreteScheduler,
    "heun": diffusers.schedulers.DPMSolverMultistepScheduler,
    "dpm++": diffusers.schedulers.DPMSolverMultistepScheduler,
    "dpm": diffusers.schedulers.DPMSolverMultistepScheduler,
    "pndm": diffusers.schedulers.PNDMScheduler,
}
