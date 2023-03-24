import time

from api.models.diffusion import ImageGenerationOptions, ImageGenerationResult
from modules.model import StableDiffusionModel


class BaseRunner:
    def __init__(self, model: StableDiffusionModel) -> None:
        self.loading = True
        self.model = model

    def activate(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def wait_loading(self):
        while self.loading:
            time.sleep(0.5)

    def generate(self, opts: ImageGenerationOptions) -> ImageGenerationResult:
        pass
