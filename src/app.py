import gc
import asyncio
import os
import sys
from io import BytesIO
from time import time
from datetime import datetime
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import ray
import torch
import numpy as np
from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

sys.path.insert(0, os.path.abspath('src'))
from background_remover.ray_bg_remover import RayBGRemoverProcessor
from background_remover.bg_removers.ben2_bg_remover import Ben2BGRemover
from background_remover.bg_removers.birefnet_bg_remover import BiRefNetBGRemover
from background_remover.image_selector import ImageSelector
from config import get_config


# Setting up default attention backend for trellis generator: can be 'flash-attn' or 'xformers'
os.environ['ATTN_BACKEND'] = 'flash-attn'


def save_image(image: Image.Image, postfix: str) -> None:
    debug = get_config("save_image") or 0
    """ Function for saving the image. """
    if debug == 0:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H-%M-%S")
    folder_path = f"/workspace/save/{today}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image.save(os.path.join(folder_path, f"{timestamp}_{postfix}.png"))


executor = ThreadPoolExecutor(max_workers=1)


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    """ Function that loading all models and warming up the generation."""
    major, _ = torch.cuda.get_device_capability(0)

    if major == 9:
        vllm_flash_attn_backend = "FLASH_ATTN"
    else:
        vllm_flash_attn_backend = "FLASHINFER"

    try:
        logger.info("Loading models...")
        # Load Trellis generator
        gen3d_model = get_config("gen3d_model")
        if gen3d_model == "trellis":
            from trellis_generator.trellis_gs_processor import GaussianProcessor
            app.state.trellis_generator = GaussianProcessor()
        elif gen3d_model == "vggt":
            from trellis_improve.trellis_vggt_gs_processor import TrellisVGGTGaussianProcessor
            app.state.trellis_generator = TrellisVGGTGaussianProcessor()

        app.state.trellis_generator.load_models()

        # Load background removers
        app.state.bg_removers_workers = [
            RayBGRemoverProcessor.remote(Ben2BGRemover),
            RayBGRemoverProcessor.remote(BiRefNetBGRemover),
        ]

        # Load image selector
        image_shape = (int(1024 / 2), int(1024 / 2), 3)
        app.state.vlm_image_selector = ImageSelector(3, image_shape, vllm_flash_attn_backend)
        app.state.vlm_image_selector.load_model()

        # Load renderer for 2x2 grid rendering
        app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if get_config("save_render"):
            from renderers.gs_renderer.renderer import Renderer
            from renderers.ply_loader import PlyLoader

            app.state.renderer = Renderer()
            app.state.ply_loader = PlyLoader()

        # Load Qwen edit
        from qma.qwen_ma import QwenMultiAngle
        app.state.qma = QwenMultiAngle()


        clean_vram()
        logger.info("Model loading is complete.")
    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        raise SystemExit("Model failed to load → exiting server")

    try:
        logger.info("Warming up Trellis generator...")
        app.state.trellis_generator.warmup_generator()
        clean_vram()
        logger.info("Warm-up is complete. Server is ready.")

    except Exception as e:
        logger.exception(f"Exception during warming up the generator: {e}")
        raise SystemExit("Warm-up failed → exiting server")

    yield


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


def remove_background(image: Image.Image, seed: int, prefix: str = "") -> Image.Image:
    """ Function for removing background from the image. """
    num_bg_remove = get_config("num_bg_remove")
    futurs = [worker.run.remote(image) for worker in app.state.bg_removers_workers[:num_bg_remove]]
    results = ray.get(futurs)

    image0 = results[0]
    output_image = results[0]
    image_ind = 0
    if len(results) > 1:
        image0 = results[0]
        image1 = results[1]
        output_image, image_ind = app.state.vlm_image_selector.select_with_image_selector(image0, image1, image, seed)

    save_image(image, f"{prefix}")
    # rename picked image
    if image_ind == 0:
        save_image(image0, f"{prefix}_bgrm_0_win")
        if num_bg_remove > 1:
            save_image(image1, f"{prefix}_bgrm_1")
    elif image_ind == 1:
        save_image(image0, f"{prefix}_bgrm_0")
        if num_bg_remove > 1:
            save_image(image1, f"{prefix}_bgrm_1_win")
    else:
        save_image(image0, f"{prefix}_bgrm_0")
        if num_bg_remove > 1:
            save_image(image1, f"{prefix}_bgrm_1")
    return output_image


def _generate_multiview_images(
    prompt_image: Image.Image,
    seed: int = -1,
    rotate_deg: float = 90.0,
    move_forward: float = 0.0,
    vertical_tilt: float = 0.0,
    wideangle: bool = False
) -> list[Image.Image]:
    n_views = get_config("num_views")
    if n_views == 0: return []
    """ Function for generating multiview images using QwenMultiAngle """
    logger.info(f"Generating {n_views} multiview images...")

    # Generate multiview images
    multiview_images = app.state.qma.infer_repeat_images(
        image=prompt_image,
        rotate_deg=rotate_deg,
        move_forward=move_forward,
        vertical_tilt=vertical_tilt,
        wideangle=wideangle,
        seed=seed if seed != -1 else 0,
        randomize_seed=(seed == -1),
        n_views=n_views
    )

    return multiview_images


def generation_block(
    prompt_image: Image.Image,
    seed: int = -1
) -> BytesIO:
    """ Function for 3D data generation using provided image"""

    if get_config("enhance_image") > 0: # enhance mode
        save_image(prompt_image, "raw")
        t_enhance_start = time()
        enhanced_image = app.state.qma.enhance_image(prompt_image, seed)
        if enhanced_image is not None:
            prompt_image = enhanced_image
            t_enhance_end = time()
            logger.debug(f"Enhance image took: {(t_enhance_end - t_enhance_start):.3f} secs.")

    images = [prompt_image]
    # Generate multiview images if n_views > 0
    if get_config("num_views") > 0:
        # Generate multiview images
        t_multiview_start = time()
        multiview_images = _generate_multiview_images( prompt_image, seed)
        images.extend(multiview_images)
        t_multiview_end = time()
        logger.debug(f"Multiview generation took: {(t_multiview_end - t_multiview_start):.3f} secs.")


    # Remove background for all images
    images_no_bg = []
    t_bg_remove_start = time()
    for idx, img in enumerate(images):
        if idx == 0:
            prefix = "enhanced" if get_config("enhance_image") > 0 else "raw"
        else:
            prefix = f"mv{idx}"
        has_alpha = img.mode in ("LA", "RGBA", "PA")
        if not has_alpha:
            img_no_bg = remove_background(img, seed, prefix=prefix)
        else:
            img_no_bg = img
        images_no_bg.append(img_no_bg)
    t_bg_remove_end = time()
    logger.debug(f"Background removal for {len(images)} images took: {(t_bg_remove_end - t_bg_remove_start):.3f} secs.")

    # Generate 3D model
    t_3d_generation_start = time()
    buffer = None
    if len(images_no_bg) > 1:
        buffer = app.state.trellis_generator.generate_3d_from_multi_images_no_bg(images_no_bg=images_no_bg, seed=seed)
    else:
        buffer = app.state.trellis_generator.generate_3d_from_image_no_bg(image_no_bg=images_no_bg[0], seed=seed)
    t_3d_generation_end = time()
    logger.debug(f"3D Generation took: {(t_3d_generation_end - t_3d_generation_start):.3f} secs.")
    clean_vram()
    return buffer

def _save_render_image(buffer: BytesIO):
    debug = get_config("save_render")
    """ Function for rendering 2x2 grid directly using Renderer instance. """
    if debug == 0:
        return


    # Render constants (from render_2x2_grid.py)
    VIEWS_NUMBER = 16
    THETA_ANGLES = np.linspace(0, 360, num=VIEWS_NUMBER)
    PHI_ANGLES = np.full_like(THETA_ANGLES, -15.0)
    GRID_VIEW_INDICES = [1, 5, 9, 13]  # 4 views for 2x2 grid
    IMG_WIDTH = 518
    IMG_HEIGHT = 518
    GRID_VIEW_GAP = 5
    CAM_RAD = 2.5
    CAM_FOV_DEG = 49.1
    REF_BBOX_SIZE = 1.5

    try:
        t_render_start = time()

        # Load PLY from buffer
        buffer.seek(0)  # Reset buffer position
        gs_data = app.state.ply_loader.from_buffer(buffer)
        gs_data = gs_data.send_to_device(app.state.device)

        # Extract only the 4 grid view angles
        theta_angles = THETA_ANGLES[GRID_VIEW_INDICES].astype(np.float32)
        phi_angles = PHI_ANGLES[GRID_VIEW_INDICES].astype(np.float32)

        # Render the 4 views with white background
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(app.state.device)
        images = app.state.renderer.render_gs(
            gs_data,
            views_number=4,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            theta_angles=theta_angles,
            phi_angles=phi_angles,
            cam_rad=CAM_RAD,
            cam_fov=CAM_FOV_DEG,
            ref_bbox_size=REF_BBOX_SIZE,
            bg_color=bg_color,
        )

        # Combine into 2x2 grid
        row_width = IMG_WIDTH * 2 + GRID_VIEW_GAP
        column_height = IMG_HEIGHT * 2 + GRID_VIEW_GAP
        combined_image = Image.new("RGB", (row_width, column_height), color="black")

        pil_images = [Image.fromarray(img.detach().cpu().numpy()) for img in images]

        combined_image.paste(pil_images[0], (0, 0))
        combined_image.paste(pil_images[1], (IMG_WIDTH + GRID_VIEW_GAP, 0))
        combined_image.paste(pil_images[2], (0, IMG_HEIGHT + GRID_VIEW_GAP))
        combined_image.paste(pil_images[3], (IMG_WIDTH + GRID_VIEW_GAP, IMG_HEIGHT + GRID_VIEW_GAP))

        # Save the rendered image
        save_image(combined_image, "render")

        # Clean up GPU memory
        del gs_data
        del images
        if app.state.device.type == 'cuda':
            torch.cuda.empty_cache()

        t_render_end = time()
        logger.debug(f"2x2 Grid rendering took: {(t_render_end - t_render_start):.3f} secs.")

    except Exception as e:
        logger.exception(f"Exception during render: {e}")
        if app.state.device.type == 'cuda':
            torch.cuda.empty_cache()

@app.post("/generate")
async def generate_model(
    prompt_image_file: UploadFile = File(...),
    seed: int = Form(default=-1)
) -> Response:
    """ Generates a 3D model as a PLY buffer """
    t_start = time()
    logger.info(f"Task received. Prompt-Image (multiview={get_config('num_views')})")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(
        executor,
        generation_block,
        prompt_image,
        seed
    )
    t_end = time()
    logger.info(f"Task completed. Time taken: {t_end - t_start:.3f} secs.")
    _save_render_image(buffer)
    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    """ Return if the server is alive """
    return {"status": "healthy"}

if __name__ == "__main__":
    import argparse
    import uvicorn

    def get_args() -> argparse.Namespace:
        """ Function for getting arguments """
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=10006)
        return parser.parse_args()

    if __name__ == "__main__":
        args: argparse.Namespace  = get_args()
        uvicorn.run(app, host=args.host, port=args.port, reload=False)
