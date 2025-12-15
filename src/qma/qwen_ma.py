from io import BytesIO
import random
from typing import Optional, Tuple, List
from time import time
import numpy as np
import torch
from PIL import Image
import sys
import os
from loguru import logger
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qma.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qma.qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

MAX_SEED = np.iinfo(np.int32).max
class QwenMultiAngle:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            transformer=QwenImageTransformer2DModel.from_pretrained(
                "linoyts/Qwen-Image-Edit-Rapid-AIO",
                subfolder="transformer",
                torch_dtype=self.dtype,
                device_map="cuda",
            ),
            torch_dtype=self.dtype
        ).to(self.device)

        # self.pipe.load_lora_weights(
        #     "dx8152/Qwen-Edit-2509-Multiple-angles",
        #     weight_name="镜头转换.safetensors",
        #     adapter_name="angles",
        # )
        # self.pipe.set_adapters(["angles"], adapter_weights=[1.0])
        # self.pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
        # self.pipe.unload_lora_weights()

        self.load_lora_weights()
        self.pipe.transformer.__class__ = QwenImageTransformer2DModel

    def load_lora_weights(self):
        if hasattr(self, 'loaded_lora') and self.loaded_lora:
            return
        t0 = time()
        self.pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles",
            weight_name="镜头转换.safetensors",
            adapter_name="angles",
        )
        self.pipe.set_adapters(["angles"], adapter_weights=[1.0])
        self.loaded_lora = True
        logger.info(f"LoRA weights loaded in {time() - t0:.2f} seconds.")
        

    def unload_lora(self):
        if not hasattr(self, 'loaded_lora') or not self.loaded_lora:
            return
        t0 = time()
        self.pipe.unload_lora_weights()
        self.loaded_lora = False
        logger.info(f"LoRA weights unloaded in {time() - t0:.2f} seconds.")

    def _build_camera_prompt(
        self,
        rotate_deg: float = 0.0,
        move_forward: float = 0.0,
        vertical_tilt: float = 0.0,
        wideangle: bool = False,
    ) -> str:
        """
        Build a camera movement prompt based on the chosen controls.

        This converts the provided control values into a prompt instruction with the corresponding trigger words for the multiple-angles LoRA.

        Args:
            rotate_deg (float, optional):
                Horizontal rotation in degrees. Positive values rotate left,
                negative values rotate right. Defaults to 0.0.
            move_forward (float, optional):
                Forward movement / zoom factor. Larger values imply moving the
                camera closer or into a close-up. Defaults to 0.0.
            vertical_tilt (float, optional):
                Vertical angle of the camera:
                - Negative ≈ bird's-eye view
                - Positive ≈ worm's-eye view
                Defaults to 0.0.
            wideangle (bool, optional):
                Whether to switch to a wide-angle lens style. Defaults to False.

        Returns:
            str:
                A text prompt describing the camera motion. If no controls are
                active, returns `"no camera movement"`.
        """
        prompt_parts = []

        # Rotation
        if rotate_deg != 0:
            direction = "left" if rotate_deg > 0 else "right"
            if direction == "left":
                prompt_parts.append(
                    f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left."
                )
            else:
                prompt_parts.append(
                    f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right."
                )

        # Move forward / close-up
        if move_forward > 5:
            prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
        elif move_forward >= 1:
            prompt_parts.append("将镜头向前移动 Move the camera forward.")

        # Vertical tilt
        if vertical_tilt <= -1:
            prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
        elif vertical_tilt >= 1:
            prompt_parts.append(
                "将相机切换到仰视视角 Turn the camera to a worm's-eye view."
            )

        # Lens option
        if wideangle:
            prompt_parts.append(" 将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

        final_prompt = " ".join(prompt_parts).strip()
        return final_prompt if final_prompt else "no camera movement"

    def infer_single_image(
        self,
        image: Image.Image,
        rotate_deg: float = 0.0,
        move_forward: float = 0.0,
        vertical_tilt: float = 0.0,
        wideangle: bool = False,
        seed: int = 0,
        randomize_seed: bool = False,
        true_guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        height: Optional[int] = None,
        width: Optional[int] = None,
        prev_output: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, int, str]:
        """
        Edit the camera angles/view of an image with Qwen Image Edit 2509 and dx8152's Qwen-Edit-2509-Multiple-angles LoRA.

        Applies a camera-style transformation (rotation, zoom, tilt, lens)
        to an input image.

        Args:
            image (PIL.Image.Image | None, optional):
                Input image to edit. If `None`, the function will instead try to
                use `prev_output`. At least one of `image` or `prev_output` must
                be available. Defaults to None.
            rotate_deg (float, optional):
                Horizontal rotation in degrees (-90, -45, 0, 45, 90). Positive values rotate
                to the left, negative to the right. Defaults to 0.0.
            move_forward (float, optional):
                Forward movement / zoom factor (0, 5, 10). Higher values move the
                camera closer; values >5 switch to a close-up style. Defaults to 0.0.
            vertical_tilt (float, optional):
                Vertical tilt (-1 to 1). -1 ≈ bird's-eye view, +1 ≈ worm's-eye view.
                Defaults to 0.0.
            wideangle (bool, optional):
                Whether to use a wide-angle lens style. Defaults to False.
            seed (int, optional):
                Random seed for the generation. Ignored if `randomize_seed=True`.
                Defaults to 0.
            randomize_seed (bool, optional):
                If True, a random seed (0..MAX_SEED) is chosen per call.
                Defaults to True.
            true_guidance_scale (float, optional):
                CFG / guidance scale controlling prompt adherence.
                Defaults to 1.0 since the demo is using a distilled transformer for faster inference.
            num_inference_steps (int, optional):
                Number of inference steps. Defaults to 4.
            height (int, optional):
                Output image height. Must typically be a multiple of 8.
                If set to 0, the model will infer a size. Defaults to 1024 if none is provided.
            width (int, optional):
                Output image width. Must typically be a multiple of 8.
                If set to 0, the model will infer a size. Defaults to 1024 if none is provided.
            prev_output (PIL.Image.Image | None, optional):
                Previous output image to use as input when no new image is uploaded.
                Defaults to None.

        Returns:
            Tuple[PIL.Image.Image, int, str]:
                - The edited output image.
                - The actual seed used for generation.
                - The constructed camera prompt string.
        """
        prompt = self._build_camera_prompt(rotate_deg, move_forward, vertical_tilt, wideangle)
        print(f"Generated Prompt: {prompt}")

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Choose input image (prefer uploaded, else last output)
        pil_images = []
        if image is not None:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            elif hasattr(image, "name"):
                pil_images.append(Image.open(image.name).convert("RGB"))
        elif prev_output:
            pil_images.append(prev_output.convert("RGB"))

        if len(pil_images) == 0:
            raise Exception("Please upload an image first.")

        if prompt == "no camera movement":
            return image, seed, prompt

        result = self.pipe(
            image=pil_images,
            prompt=prompt,
            height=height if height != 0 else None,
            width=width if width != 0 else None,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]

        return result, seed, prompt

    def infer_repeat_images(
        self,
        image: Image.Image,
        rotate_deg: float = 0.0,
        move_forward: float = 0.0,
        vertical_tilt: float = 0.0,
        wideangle: bool = False,
        seed: int = 0,
        randomize_seed: bool = False,
        n_views: int = 1,
    ) -> List[Image.Image]:
        """
        Edit the camera angles/view of an image with Qwen Image Edit 2509 and dx8152's Qwen-Edit-2509-Multiple-angles LoRA.

        Applies a camera-style transformation (rotation, zoom, tilt, lens)
        to an input image.
        """
        self.load_lora_weights()
        if n_views < 1: return []
        output = [image.copy()]
        for _ in range(n_views):
            image = output[-1]
            result, seed, prompt = self.infer_single_image(image, rotate_deg, move_forward, vertical_tilt, wideangle, seed, randomize_seed)
            output.append(result.copy())
        return output[1:] # return the results without the original image
    
    def enhance_image(self, image: Image.Image, seed: int = -1) -> Image.Image:
        """
        Perform a normal image edit without camera angle changes.
        """
        self.unload_lora()
        if seed < 0:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt = "Show this object in three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details"
        negative_prompt = "NSFW, (worst quality:2), (low quality:2), (normal quality:2), multiple objects, complex background, environmental scene, dramatic lighting, shadows, artistic style, painterly, sketchy, conceptual art, abstract, cropped object, partial view, occluded features, blurry, motion blur, depth of field, bokeh, (monochrome), (grayscale), (skin blemishes:1.331), (acne:1.331), (age spots:1.331), (extra fingers:1.61051), (deformed limbs:1.331), (malformed limbs:1.331), (ugly:1.331), (poorly drawn hands:1.5), (poorly drawn feet:1.5), (poorly drawn face:1.5), (mutated hands:1.331), (bad anatomy:1.21), (distorted face:1.331), (disfigured:1.331), (low contrast), (underexposed), (overexposed), (amateur), (blurry), (bad proportions:1.331), (extra limbs:1.331), (fused fingers:1.61051), (unclear edges:1.331)"
        edited_image = self.pipe(
            image=[image.convert("RGB")],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,
            generator=generator,
            true_cfg_scale=1.0,
            num_images_per_prompt=1,
        ).images[0]
        return edited_image
