from PIL import Image
from time import time
import ray
import torch
from loguru import logger
from background_remover.bg_removers.base_bg_remover import BaseBGRemover


@ray.remote(num_gpus=0.05)
class RayBGRemoverProcessor:
    def __init__(self,  model: BaseBGRemover):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._bg_remover = model(device)
        self._bg_remover.load_model()

    def unload_model(self):
       self._bg_remover.unload_model()
       del self._bg_remover
       self._bg_remover = None
       ray.shutdown()

    def run(self, image: Image.Image) -> Image:
        t0 = time()
        result_image = self._bg_remover.remove_bg(image)
        t1 = time()
        logger.debug(f"Background removal took: {t1 - t0} secs.")
        return result_image
