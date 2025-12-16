from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop

# from config import Settings
from loguru import logger


class BackgroundRemovalService:
    def __init__(self):
        """
        Initialize the BackgroundRemovalService.
        """
        # Set padding percentage, output size
        self.padding_percentage = 0.2
        self.output_size = (518, 518)
        self.limit_padding = True
        self.model_id = "hiepnd11/rm_back2.0"

        # Set device
        self.device = f"cuda:0" if torch.cuda.is_available() else "cpu"

        # Set model
        self.model: AutoModelForImageSegmentation | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.model_id} model...")

        # Load model
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            logger.success(f"{self.model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.model_id} model: {e}")

    def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.model_id} model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        try:
            t1 = time.time()
            # PIL.Image (H, W, C) C=3
            if image.mode == "RGBA":
                image = image.convert('RGB')

            # Tensor (H, W, C) -> (C, H',W')
            rgb_tensor = self.transforms(image).to(self.device)
            output = self._remove_background(rgb_tensor)

            image_without_background = to_pil_image(output[:3])

            removal_time = time.time() - t1
            logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}")

            return image_without_background

        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return image

    def _remove_background(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """
        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)
        input_tensor = self.normalize(image_tensor).unsqueeze(0)

        with torch.no_grad():
            # Get mask from model (1, 1, H, W)
            preds = self.model(input_tensor)[-1].sigmoid()
            # Reshape and quantize mask values: (1, 1, H, W) -> (1, H, W) -> (H, W)
            mask = preds[0].squeeze().mul_(255).int().div(255).float()

        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > 0.8)
        if len(bbox_indices) == 0:
            crop_args = dict(top = 0, left = 0, height = mask.shape[1], width = mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center =  (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = int(size * padded_size_factor)
            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)

            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )

            mask = mask.unsqueeze(0)
            # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
            tensor_rgba = torch.cat([image_tensor*mask, mask], dim=-3)
            output = resized_crop(tensor_rgba, **crop_args, size = self.output_size, antialias=False)
            return output
