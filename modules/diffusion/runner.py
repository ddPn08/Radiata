import time
from PIL.Image import Image


class BaseRunner:
    loading = True

    def __init__(self) -> None:
        pass

    def activate() -> None:
        pass

    def teardown() -> None:
        pass

    def wait_loading(self):
        if not self.loading:
            return
        while self.loading:
            time.sleep(0.5)

    def infer(
        self,
        prompt: str,
        negative_prompt: str,
        batch_size: int,
        batch_count: int,
        scheduler_id: str,
        steps: int,
        scale: int,
        image_height: int,
        image_width: int,
        seed: int,
    ) -> tuple[list[tuple[list[Image], dict, dict]], float]:
        pass
