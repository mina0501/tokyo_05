from io import BytesIO
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
from qwen_ma import QwenMultiAngle
import uvicorn

# --- Model Loading ---
qwen_ma = QwenMultiAngle()

app = FastAPI()


@app.post("/test")
def test_edit(
    prompt_image: UploadFile = File(...),
    rotate_deg: float = Form(0.0),
    move_forward: float = Form(0.0),
    vertical_tilt: float = Form(0.0),
    wideangle: bool = Form(False),
    n_views: int = Form(1),
):

    pil_image = Image.open(prompt_image.file)
    generated_images = qwen_ma.infer_repeat_images(
        image=pil_image,
        rotate_deg=rotate_deg,
        move_forward=move_forward,
        vertical_tilt=vertical_tilt,
        wideangle=wideangle,
        n_views=n_views,
    )

    results = [pil_image] + generated_images
    # merge results into a single image
    final_image = Image.new("RGB", (pil_image.width * len(results), pil_image.height))
    for i, pil_image in enumerate(results):
        final_image.paste(pil_image, (i * pil_image.width, 0))

    buffered = BytesIO()
    final_image.save(buffered, format="PNG")
    buffered.seek(0)
    return StreamingResponse(buffered, media_type="image/png")

@app.post("/gen_multi_angle")
def gen_multi_angle(
    prompt_image: UploadFile = File(...),
    rotate_deg: float = Form(0.0),
    move_forward: float = Form(0.0),
    vertical_tilt: float = Form(0.0),
    wideangle: bool = Form(False),
    n_views: int = Form(1),
):
    pil_image = Image.open(prompt_image.file)
    generated_images = qwen_ma.infer_repeat_images(image=pil_image, rotate_deg=rotate_deg, move_forward=move_forward, vertical_tilt=vertical_tilt, wideangle=wideangle, n_views=n_views)
    base64_images = [base64.b64encode(image.tobytes()).decode("utf-8") for image in generated_images]
    return {
        "base64_images": base64_images
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
