#working code for generating video using HunyuanVideoPipeline
import os

# ========== SET GPU ENVIRONMENT BEFORE TORCH/Diffusers ==========
gpu_id = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # Hides all other GPUs, makes GPU 2 become cuda:0

# ========== IMPORT LIBRARIES ==========
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# ========== SET DEVICE ==========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (maps to actual GPU {gpu_id})")

# ========== LOAD MODEL ==========
model_id = "Path/to/your/model"  # Update with your actual model path
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
).to(device)

pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.float16
).to(device)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

# ========== VIDEO SETTINGS ==========
video_length_seconds = 7
fps = 8
num_frames = video_length_seconds * fps

# ========== GENERATE VIDEO ==========
output = pipe(
    prompt="A man standing in a classroom, giving a presentation to a group of students. he is wearing a cream-colored long-sleeved shirt and dark blue pants, with a black belt around his waist. he has a beard and is wearing glasses. the classroom has a green chalkboard and white walls, and there are desks and chairs arranged in a semi-circle around him. the man is standing in the middle of the classroom, with his hands gesturing as he speaks. he appears to be a middle-aged man with a serious expression, and his hair is styled in a short, neat manner. the students in the classroom are of various colors, including brown, black, and white, and they are seated in front of him, facing the man in the center of the image. they are all facing the same direction and appear to be engaged in the presentation.",
    height=320,
    width=512,
    num_frames=num_frames,
    num_inference_steps=30,
).frames[0]

# ========== SAVE VIDEO ==========
export_to_video(output, "HunyuanVideo_output.mp4", fps=fps)
print("Video generation complete and saved to HunyuanVideo_output8.mp4")