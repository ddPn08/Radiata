import time
from queue import Queue
from typing import *

from api.models.diffusion import ImageGenerationOptions
from lib.diffusers.scheduler import SCHEDULERS
from modules import config
from modules.model import DiffusersModel
from modules.shared import hf_diffusers_cache_dir


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

    def yielder(self):
        results = []

        queue = Queue()
        done = object()
        total_steps = 0

        def callback(*args, **kwargs):
            nonlocal total_steps
            total_steps += 1
            queue.put((total_steps, results))

        def on_done(feature):
            queue.put(done)

        def wait():
            while True:
                item = queue.get()
                if item is done:
                    break
                yield item

        return callback, on_done, wait

    def generate(self, opts: ImageGenerationOptions):
        pass

    def get_scheduler(self, scheduler_name: str, model_id: str = None):
        return SCHEDULERS[scheduler_name].from_pretrained(
            self.model.model_id if model_id is None else model_id,
            subfolder="scheduler",
            use_auth_token=config.get("hf_token"),
            cache_dir=hf_diffusers_cache_dir(),
        )
