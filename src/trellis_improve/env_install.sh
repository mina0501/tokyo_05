pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers
pip install rembg tqdm omegaconf matplotlib opencv-python imageio jaxtyping einops 
pip install SentencePiece accelerate trimesh PyMCubes xatlas libigl
pip install git+https://github.com/facebookresearch/pytorch3d
pip install git+https://github.com/NVlabs/nvdiffrast
pip install open3d
pip install fastapi
pip install uvicorn
pip install open-clip-torch==2.23.0
pip install setuptools==69.5.1
pip install uvicorn==0.32.0
pip install onnxruntime==1.19.2
pip install typeguard==4.4.1
pip install rembg==2.0.59
pip install "numpy<2"
pip install boto3
pip install python-dotenv
pip install git+https://github.com/openai/CLIP.git
pip install aiohttp
pip install watchdog
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install plyfile pillow imageio imageio-ffmpeg tqdm easydict 
pip install  opencv-python-headless
pip install scipy ninja rembg onnxruntime trimesh xatlas pyvista 		
pip install pymeshfix igraph transformers spconv-cu118
pip install --no-build-isolation flash-attn==2.7.3
pip install git+https://github.com/404-Repo/spz.git
pip install pybase64
pip install DeepCache
pip install meshio
pip install -U bitsandbytes
pip install peft
pip install pytod
pip install pytorch_lightning
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
pip install python-multipart
pip install loguru==0.7.3
pip install kornia