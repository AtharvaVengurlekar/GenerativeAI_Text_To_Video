
import os
# ================== FORCE GPU 1 =================== #
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 becomes logical cuda:0 now

import subprocess
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# ================== CONFIG =================== #
model_id = "Path/to/your/model"  # Update with your actual model path
output_video_name = "output_generated.mp4"
final_output_video_name = "output_final.mp4"

input_frames_dir = "Path/to/your/inputs/folder"  # Update with your actual path
output_frames_dir = "Path/to/your/results/folder"  # Update with your actual path
realesrgan_model_path = "Path/to/your/RealESRGAN_x4plus.pth"  # Update with your actual path

# ================== SETUP =================== #
print("Setting Environment for GPU 1 (CUDA_VISIBLE_DEVICES=1)...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(input_frames_dir, exist_ok=True)
os.makedirs(output_frames_dir, exist_ok=True)

# ================== LOAD MODEL =================== #
print("Loading HunyuanVideo Model...")
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)

pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.float16
)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

# ================== GENERATE VIDEO =================== #
print("Generating Video...")
video_length_seconds = 7
fps = 8
num_frames = video_length_seconds * fps

output = pipe(
    prompt="A woman using a smartphone while sitting on a sofa, natural lighting, realistic",
    height=16,
    width=16,
    num_frames=num_frames,
    num_inference_steps=30,
).frames[0]

export_to_video(output, output_video_name, fps=fps)
print(f"Video generated: {output_video_name}")

# ================== EXTRACT FRAMES =================== #
print("Extracting Frames for Upscaling...")
subprocess.run(
    f"ffmpeg -i {output_video_name} -vf fps=30 {input_frames_dir}/frame%03d.png",
    shell=True,
    check=True,
)
print("Frames extracted successfully!")

# ================== UPSCALE FRAMES =================== #
subprocess.run(
    "python3 /Path/to/your/inference_realesrgan.py/file "  # Update with your actual path
    "--model_path Path/to/your/RealESRGAN_x4plus.pth "  # Update with your actual path
    f"--input {input_frames_dir} "
    "--suffix out",
    shell=True,
    check=True,
)
print("Upscaling Completed.")

# ================== CREATE FINAL UPSCALED VIDEO =================== #
print("Generating Final Upscaled Video...")
subprocess.run(
    f"ffmpeg -framerate 30 -i {output_frames_dir}/frame%03d_out.png -c:v libx264 -pix_fmt yuv420p {final_output_video_name}",
    shell=True,
    check=True,
)
print(f"Final Upscaled Video Ready: {final_output_video_name}")