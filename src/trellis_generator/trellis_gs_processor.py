import gc
from io import BytesIO
from PIL import Image

import torch

from loguru import logger
from trellis_generator.pipelines import TrellisImageTo3DPipeline
from background_remover.utils.rand_utils import secure_randint, set_random_seed
from config import get_config

class GaussianProcessor:
    """Generates 3d models from images with background removed"""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_to_3d_pipeline: TrellisImageTo3DPipeline | None = None
        self.gaussians: torch.Tensor | None = None

    def load_models(self, model_name: str = "microsoft/TRELLIS-image-large") -> None:
        """ Function for preloading Trellis model for image -> 3D pipeline """

        self._image_to_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self._image_to_3d_pipeline.to(self._device)
        torch.cuda.empty_cache()

    def unload_models(self) -> None:
        """  Function for unloading Trellis model """

        del self._image_to_3d_pipeline
        del self.gaussians

        self._image_to_3d_pipeline = None
        self.gaussians = None

        gc.collect()
        torch.cuda.empty_cache()

    def warmup_generator(self):
        """ Function for warming up the generator. """

        dummy = Image.new("RGBA", (64, 64), color=(128, 128, 128, 255))
        self.generate_3d_from_image_no_bg(image_no_bg=dummy, seed=0)

    def generate_3d_from_image_no_bg(self, image_no_bg: Image.Image, seed: int) -> BytesIO:
        """ Function for generating a 3D object using an input image without background. """

        if seed < 0:
            set_seed = secure_randint(0, 10000)
            set_random_seed(set_seed)
        else:
            set_random_seed(seed)

        outputs = self._image_to_3d_pipeline.run(
            image_no_bg,
            num_samples=3,
            sparse_structure_sampler_params={
                "steps": 8,
                "cfg_strength": 5.75,
            },
            slat_sampler_params={
                "steps": 20,
                "cfg_strength": 2.4,
            },
        )
        self.gaussians = outputs["gaussian"][0]

        buffer = BytesIO()
        self.gaussians.save_ply(buffer)
        buffer.seek(0)

        return buffer

    def generate_3d_from_multi_images_no_bg(self, images_no_bg: list[Image.Image], seed: int) -> BytesIO:
        """ Function for generating a 3D object using multiple input images without background. """

        if seed < 0:
            set_seed = secure_randint(0, 10000)
            set_random_seed(set_seed)
        else:
            set_random_seed(seed)

        outputs = self._image_to_3d_pipeline.run_multi_image(
            images_no_bg,
            mode=get_config("trellis_multi_mode"),
            num_samples=3,
            sparse_structure_sampler_params={
                "steps": 8,
                "cfg_strength": 5.75,
            },
            slat_sampler_params={
                "steps": 20,
                "cfg_strength": 2.4,
            },
        )
        self.gaussians = outputs["gaussian"][0]

        buffer = BytesIO()
        self.gaussians.save_ply(buffer)
        buffer.seek(0)

        return buffer
