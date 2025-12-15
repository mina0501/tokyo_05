import os
import numpy as np
import cv2
import importlib.resources as pkg_resources
from PIL import Image
from loguru import logger
from background_remover.config import load_vlm_image_selector_settings_from_yaml
from background_remover.image_selector_vlm.vlm_image_selector import VLMImageSelector
from background_remover.utils.image_utils import recenter_image, resize_image


class ImageSelector:
    def __init__(
            self,
            image_number_in_use: int,
            image_shape: tuple[int, int, int],
            flash_attn_backend: str = "FLASHINFER",
            kv_cache_dtype: str = "fp8_e4m3"
    ) -> None:
        self._image_shape = image_shape
        config_path = pkg_resources.files("background_remover.configs").joinpath(
            "vlm_image_selector_config.yml"
        )
        vlm_settings = load_vlm_image_selector_settings_from_yaml(config_path)
        self._vlm_image_selector = VLMImageSelector(
            vlm_settings=vlm_settings,
            image_number_in_use=image_number_in_use,
            image_shape=image_shape,
            kv_cache_dtype=kv_cache_dtype
        )
        os.environ["VLLM_ATTENTION_BACKEND"] = flash_attn_backend

    def load_model(self, custom_mem_util: float | None = None) -> None:
        self._vlm_image_selector.load_model(custom_mem_util)

    def unload_model(self) -> None:
        self._vlm_image_selector.unload_model()

    def _is_same_by_iou(self, image1: Image.Image, image2: Image.Image) -> bool:
        """
        Function that checks if the two images are similar by calculating the IOU mask
        """
        cv_image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
        cv_image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)

        mask1 = cv2.threshold(cv_image1, 1, 255, cv2.THRESH_BINARY)[1]
        mask2 = cv2.threshold(cv_image2, 1, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((3, 3), np.uint8)

        mask1_clean = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask2_clean = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)

        # cv2.imwrite("/workspace/save/mask1.png", mask1)
        # cv2.imwrite("/workspace/save/mask2.png", mask2)
        # cv2.imwrite("/workspace/save/mask1_clean.png", mask1_clean)
        # cv2.imwrite("/workspace/save/mask2_clean.png", mask2_clean)

        union = np.sum(mask1_clean | mask2_clean)

        if union == 0:
            logger.warning("Both images have no foreground content. Considering them as same.")
            return True

        intersection = np.sum(mask1_clean & mask2_clean)
        iou = intersection / union

        logger.warning(f"IOU: {iou:.4f} (intersection: {intersection}, union: {union})")
        return iou > 0.93

    def select_with_image_selector(
            self, image1: Image.Image, image2: Image.Image, reference_image: Image.Image, seed: int
    ) -> tuple[Image.Image, int]:
        """
        Function that process three input images (two results from BG removers and one reference generated image) and
        select one result using image selector.
        """

        self._vlm_image_selector.create_sampling_params(seed)

        image_list = [image1, image2]

        image1_resized = resize_image(image_list[0], self._image_shape[0], self._image_shape[1])
        image2_resized = resize_image(image_list[1], self._image_shape[0], self._image_shape[1])
        image_ref = resize_image(reference_image, self._image_shape[0], self._image_shape[1])

        image_list_resized = [image1_resized, image_ref, image2_resized]

        if self._is_same_by_iou(image1_resized, image2_resized):
            logger.warning("Images are similar by IOU, using image 1")
            image_ind = 0
        else:
            logger.warning("Images are NOT similar by IOU, using image selector")
            image_ind = self._vlm_image_selector.select(image_list_resized)

        image_no_bg = image_list[image_ind]

        output_img = recenter_image(image_no_bg, reference_image.size[0], 0.0)
        return output_img, image_ind
