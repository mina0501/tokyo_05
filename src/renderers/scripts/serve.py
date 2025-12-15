#!/usr/bin/env python3
"""FastAPI service for rendering 2x2 grid from uploaded splat/ply files"""
from time import time
import os
import sys
import io
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from renderers.gs_renderer.renderer import Renderer
from renderers.ply_loader import PlyLoader
from render_2x2_grid import (
    splat_to_ply_bytes,
    render_splat_grid,
    VIEWS_NUMBER,
    THETA_ANGLES,
    PHI_ANGLES,
    GRID_VIEW_INDICES,
    IMG_WIDTH,
    IMG_HEIGHT,
    GRID_VIEW_GAP,
    CAM_RAD,
    CAM_FOV_DEG,
    REF_BBOX_SIZE,
)

# Global state for renderer and loader
class AppState:
    device: torch.device = None
    ply_loader: PlyLoader = None
    renderer: Renderer = None

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize renderer on startup, cleanup on shutdown"""
    # Startup
    print("Initializing renderer...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering")

    app_state.device = torch.device("cuda")
    app_state.ply_loader = PlyLoader()
    app_state.renderer = Renderer()
    print(f"Renderer initialized on device: {app_state.device}")

    yield

    # Shutdown
    print("Cleaning up...")
    if app_state.device.type == 'cuda':
        torch.cuda.empty_cache()
    print("Shutdown complete")


app = FastAPI(
    title="2x2 Grid Renderer API",
    description="Upload .splat or .ply files to render 2x2 grid views",
    version="1.0.0",
    lifespan=lifespan,
)


def render_uploaded_file(file_bytes: bytes, filename: str) -> Image.Image:
    """Render uploaded file bytes to 2x2 grid image"""
    # Determine file type
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ['.splat', '.ply']:
        raise ValueError(f"Unsupported file type: {file_ext}. Only .splat and .ply are supported.")

    # Convert to PLY bytes if needed
    if file_ext == '.splat':
        ply_bytes = splat_to_ply_bytes(file_bytes)
    else:
        ply_bytes = file_bytes

    ply_buffer = io.BytesIO(ply_bytes)

    # Load PLY from buffer
    gs_data = app_state.ply_loader.from_buffer(ply_buffer)
    gs_data = gs_data.send_to_device(app_state.device)

    # Extract only the 4 grid view angles
    theta_angles = THETA_ANGLES[GRID_VIEW_INDICES].astype(np.float32)
    phi_angles = PHI_ANGLES[GRID_VIEW_INDICES].astype(np.float32)

    # Render the 4 views with white background
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(app_state.device)
    images = app_state.renderer.render_gs(
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

    # Clean up GPU memory
    del gs_data
    del images
    if app_state.device.type == 'cuda':
        torch.cuda.empty_cache()

    return combined_image


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "2x2 Grid Renderer",
        "device": str(app_state.device),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/render")
async def render_2x2_grid(file: UploadFile = File(...)):
    """
    Upload a .splat or .ply file and get back a 2x2 grid render as PNG

    Args:
        file: Uploaded .splat or .ply file

    Returns:
        StreamingResponse with PNG image
    """
    t0 = time()
    # Validate file extension
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ['.splat', '.ply']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Only .splat and .ply files are supported."
        )

    try:
        # Read uploaded file
        file_bytes = await file.read()

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Render to 2x2 grid
        grid_image = render_uploaded_file(file_bytes, filename)

        # Convert PIL Image to bytes for streaming
        img_byte_arr = io.BytesIO()
        grid_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        t1 = time()
        print(f"Time taken: {t1 - t0} secs.")
        # Return as streaming response
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename={Path(filename).stem}_2x2_grid.png"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Clean up GPU memory on error
        if app_state.device and app_state.device.type == 'cuda':
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "renderer_initialized": app_state.renderer is not None,
        "ply_loader_initialized": app_state.ply_loader is not None,
        "device": str(app_state.device) if app_state.device else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


if __name__ == "__main__":
    import uvicorn

    # Set CUDA environment variables before importing torch/gsplat
    if 'CONDA_PREFIX' in os.environ:
        conda_prefix = os.environ['CONDA_PREFIX']
        nvcc_path = os.path.join(conda_prefix, 'bin', 'nvcc')
        if os.path.exists(nvcc_path):
            os.environ['CUDA_HOME'] = conda_prefix
            os.environ['CUDACXX'] = nvcc_path
            os.environ['CUDA_INCLUDE_PATH'] = f"{conda_prefix}/include:{conda_prefix}/targets/x86_64-linux/include"
            os.environ['CPATH'] = f"{conda_prefix}/include:{conda_prefix}/targets/x86_64-linux/include" + (os.pathsep + os.environ.get('CPATH', '') if os.environ.get('CPATH') else '')
            os.environ['PATH'] = os.path.join(conda_prefix, 'bin') + os.pathsep + os.environ.get('PATH', '')

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to prevent CUDA issues
        log_level="info"
    )
