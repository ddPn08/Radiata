import time
from typing import Dict, List, Tuple

from PIL.Image import Image


class BaseRunner:
    loading = True

    def __init__(self) -> None:
        pass

    def activate(self) -> None:
        pass

    def teardown(self) -> None:
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
    ) -> Tuple[List[Tuple[List[Image], Dict, Dict]], float]:
        pass
