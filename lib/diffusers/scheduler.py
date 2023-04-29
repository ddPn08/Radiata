import diffusers

SCHEDULERS = {
    "ddim": diffusers.DDIMScheduler,
    "deis": diffusers.DEISMultistepScheduler,
    "dpm2": diffusers.KDPM2DiscreteScheduler,
    "dpm2-a": diffusers.KDPM2AncestralDiscreteScheduler,
    "euler_a": diffusers.EulerAncestralDiscreteScheduler,
    "euler": diffusers.EulerDiscreteScheduler,
    "heun": diffusers.DPMSolverMultistepScheduler,
    "dpm++": diffusers.DPMSolverMultistepScheduler,
    "dpm": diffusers.DPMSolverMultistepScheduler,
    "pndm": diffusers.PNDMScheduler,
}
