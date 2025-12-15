import gc
from io import BytesIO
import torch
import numpy as np
from PIL import Image

from config import get_config
from trellis_improve.trellis.pipelines import TrellisVGGTTo3DPipeline
from background_remover.utils.rand_utils import secure_randint, set_random_seed
from trellis_improve.trellis.representations.gaussian import Gaussian

class TrellisVGGTGaussianProcessor:
    """Generates 3d models and videos"""

    def __init__(self):
        self._device = torch.device("cuda")
        self._image_to_3d_pipeline: TrellisVGGTTo3DPipeline | None = None

    def load_models(self, model_name: str = "Stable-X/trellis-vggt-v0-2"):
        self._image_to_3d_pipeline = TrellisVGGTTo3DPipeline.from_pretrained(model_name)
        self._image_to_3d_pipeline.to(self._device)
        self._image_to_3d_pipeline.VGGT_model.to(self._device)
        torch.cuda.empty_cache()

    def warmup_generator(self):
        """ Function for warming up the generator. """

        dummy = Image.new("RGBA", (64, 64), color=(128, 128, 128, 255))
        self.generate_3d_from_image_no_bg(image_no_bg=dummy, seed=0)

    def unload_model(self):
        del self._image_to_3d_pipeline
        self._image_to_3d_pipeline = None

        torch.cuda.empty_cache()
        gc.collect()

    def generate_3d_from_image_no_bg(self, image_no_bg: Image.Image, seed: int) -> BytesIO:
        """ Function for generating a 3D object using an input image without background. """

        if seed < 0:
            set_seed = secure_randint(0, 10000)
            set_random_seed(set_seed)
        else:
            set_random_seed(seed)

        outputs = self._image_to_3d_pipeline.run(
            [image_no_bg],
            num_samples=1,
            sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 25, "cfg_strength": 3},
            mode=get_config("trellis_multi_mode"),
        )
        self.gaussians = outputs["gaussian"][0]

        buffer = BytesIO()
        self.gaussians.save_ply(buffer)
        buffer.seek(0)

        return buffer

    def generate_3d_from_multi_images_no_bg(self, images_no_bg: list[Image.Image], seed: int) -> BytesIO:
        """
        Generates a 3D model in PLY format
        Args:
            images_no_bg: a list of PIL images
            seed: random seed for trellis model generator
        Returns:
            BytesIO: The Buffer object containing the 3D model in PLY format
        """
        if seed < 0:
            set_seed = secure_randint(0, 10000)
            set_random_seed(set_seed)
        else:
            set_random_seed(seed)

        images_no_bg = [self._image_to_3d_pipeline.preprocess_image(image) for image in images_no_bg]

        outputs = self._image_to_3d_pipeline.run(
            images_no_bg,
            num_samples=1,
            sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 25, "cfg_strength": 3},
            mode=get_config("trellis_multi_mode"),
        )

        base_gaussian: Gaussian = outputs["gaussian"][0]

        buffer = BytesIO()
        base_gaussian.save_ply(buffer)
        buffer.seek(0)

        return buffer
