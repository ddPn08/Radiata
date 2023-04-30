import time
from typing import *

from api.models.diffusion import ImageGenerationOptions
from lib.diffusers.scheduler import SCHEDULERS
from modules import config
from modules.model import DiffusersModel
from modules.shared import hf_cache_dir


class BaseRunner:
    def __init__(self, model: DiffusersModel) -> None:
        self.loading = None
        self.model = model

    def activate(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def wait_loading(self):
        if self.loading is None:
            self.activate()
        while self.loading:
            time.sleep(0.5)

    def generate(self, opts: ImageGenerationOptions):
        pass

    def get_scheduler(self, scheduler_name: str):
        return SCHEDULERS[scheduler_name].from_pretrained(
            self.model.model_id,
            subfolder="scheduler",
            use_auth_token=config.get("hf_token"),
            cache_dir=hf_cache_dir(),
        )
