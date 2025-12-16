conda create -n qwen_angle python=3.11
conda activate qwen_angle
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128

hf download Qwen/Qwen-Image-Edit-2506
hf download linoyts/Qwen-Image-Edit-Rapid-AIO
hf download dx8152/Qwen-Edit-2509-Multiple-angles --local-dir /root/dx8152/Qwen-Edit-2509-Multiple-angles
