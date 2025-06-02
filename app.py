from flask import Flask, request, jsonify
import subprocess
import os
import torch
import boto3
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from threading import Thread
import time
import logging
import random
import string
from pymongo import MongoClient
from datetime import datetime
import pytz
from dotenv import load_dotenv
load_dotenv()

 
app = Flask(__name__)
 
# ================== CONFIG =================== #
 
model_id = os.getenv('MODEL_ID')
input_frames_dir = "Path/to/your/inputs/folder"  # Update with your actual path
output_frames_dir = "Path/to/your/results/folder"  # Update with your actual path
realesrgan_model_path = "Path/to/your/RealESRGAN_x4plus.pth"  # Update with your actual path
device_id = int(os.getenv("CUDA_DEVICE_ID", 0))
video_status_dict = {}
session_history = {}
 
# AWS S3 Configuration
aws_access_key = os.getenv('AWS_ACCESS_KEY')
aws_secret_key = os.getenv('AWS_SECRET_KEY')
bucket_name = os.getenv('BUCKET_NAME')
region_name = os.getenv('REGION_NAME')

print("AWS_ACCESS_KEY:", aws_access_key)
print("AWS_SECRET_KEY:", aws_secret_key)
print("REGION_NAME:", region_name)
print("BUCKET_NAME:", bucket_name)

 
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name
)
 
# MongoDB Setup
mongo_client = MongoClient(os.getenv('MONGODB_URI'))
db = mongo_client['text_to_video_chatbot_logs']
api_requests_collection = db['chat_logs']
 
# Logger Setup (IST Timezone)
IST = pytz.timezone('Asia/Kolkata')
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")
 
formatter = ISTFormatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("server.log", mode="w")  # new log created
file_handler.setFormatter(formatter)
 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
 
 
# Environment Setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

 
DEFAULT_HEIGHT = 544
DEFAULT_WIDTH = 960
MAX_HEIGHT = 720
MAX_WIDTH = 1280
 
# Load Model Only Once
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.float16
)
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()
 
# Upload to S3
def upload_to_s3(file_path, bucket, folder):
    try:
        s3_key = f"{folder}/{os.path.basename(file_path)}"
        with open(file_path, 'rb') as file_data:
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=file_data,
                ContentType='video/mp4'
            )
        url = f"https://{bucket}.s3.{region_name}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.exception("S3 upload failed")
        return None
 
# Process Video Generation
def process_video(session_id, reference_id, prompt, height, width, num_frames, fps, num_inference_steps):
    video_id = str(int(time.time()))
    video_filename = f"{session_id}_{reference_id}_generated.mp4"
    final_video_filename = f"{session_id}_{reference_id}_final_upscaled.mp4"
 
    video_status_dict[video_id] = "processing"
    if session_id not in session_history:
        session_history[session_id] = []
 
    session_entry = {
        "reference_id": reference_id,
        "status": "processing",
        "prompt": prompt,
        "height": height,
        "width": width,
        "s3_url": None,
        "session_id": session_id,
    }
    session_history[session_id].append(session_entry)
 
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
 
    # video_length_seconds = 7
    # fps = 8
    # num_frames = video_length_seconds * fps
 
    print("Generating Video...")
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    ).frames[0]
 
    export_to_video(output, video_filename, fps=fps)
    print(f"Video generated: {video_filename}")
 
    subprocess.run(f"ffmpeg -i {video_filename} -vf fps=30 {input_frames_dir}/frame%03d.png", shell=True, check=True)
    subprocess.run(f"python3 /app/finalpipeline/inference_realesrgan.py --model_path {realesrgan_model_path} --input {input_frames_dir} --suffix out", shell=True, check=True)
    subprocess.run(f"ffmpeg -framerate 30 -i {output_frames_dir}/frame%03d_out.png -c:v libx264 -pix_fmt yuv420p {final_video_filename}", shell=True, check=True)
 
    s3_url = upload_to_s3(final_video_filename, bucket_name, folder="video_outputs")
    print(f"DEBUG: S3 URL = {s3_url}")

    # Ensure the status is updated to "completed" once the video is uploaded
    for entry in session_history[session_id]:
        if entry["reference_id"] == reference_id:
            entry["status"] = "completed"
            entry["s3_url"] = s3_url
 
    video_status_dict[video_id] = "completed"
 
    log_data = {
        "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")),
        "method": "POST",
        "endpoint": "/generate_video",
        "session_id": session_id,
        "reference_id": reference_id,
        "prompt": prompt,
        "height": height,
        "width": width,
        "fps": fps,
        "num_frames": num_frames,
        "video_url": s3_url,
    }
    api_requests_collection.insert_one(log_data)
    logger.info(str(log_data))
 
@app.route('/generate_video', methods=['POST'])
def generate_video():
    data = request.get_json()
    prompt = data.get('prompt', 'Default Prompt')
    height = data.get('height', DEFAULT_HEIGHT)
    width = data.get('width', DEFAULT_WIDTH)
    num_frames = data.get('num_frames', 56)
    fps = data.get('fps', 8)
    num_inference_steps = data.get('num_inference_steps', 30)
    session_id = data.get('session_id') or ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
    reference_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
 
    # Add session to history before starting the video generation
    session_entry = {
        "reference_id": reference_id,
        "status": "processing",
        "prompt": prompt,
        "height": height,
        "width": width,
        "s3_url": None,
        "session_id": session_id,
    }
 
    if session_id not in session_history:
        session_history[session_id] = []
 
    session_history[session_id].append(session_entry)
 
    # Start video generation in a separate thread
    def task():
        process_video(session_id, reference_id, prompt, height, width, num_frames, fps, num_inference_steps)
 
    thread = Thread(target=task)
    thread.start()
 
    # Return session_id and reference_id immediately
    return jsonify({
        "session_id": session_id,
        "reference_id": reference_id,
    })
 
@app.route("/get_status", methods=["POST"])
def get_status():
    data = request.get_json()
    session_id = data.get("session_id")
    reference_id = data.get("reference_id")
 
    if not session_id or not reference_id:
        return jsonify({"error": "Missing session_id or reference_id"}), 400
 
    history = session_history.get(session_id, [])
    for entry in history:
        if entry["reference_id"] == reference_id:
            return jsonify({"reference_id": reference_id, "status": entry["status"]}), 200
 
    return jsonify({"reference_id": reference_id, "status": "Invalid reference_id"}), 404
 
@app.route("/download_video/<reference_id>", methods=["POST"])
def download_video(reference_id):
    data = request.get_json()
    session_id = data.get("session_id")
 
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
 
    history = session_history.get(session_id, [])
    for entry in history:
        if entry["reference_id"] == reference_id:
            if entry.get("session_id") != session_id:
                return jsonify({"error": "Reference ID does not belong to this session"}), 403
 
            # Return the correct S3 URL now
            return jsonify({
                "s3_url": entry["s3_url"]
            }), 200
 
    return jsonify({"error": "Invalid reference_id"}), 404
 
@app.route("/history", methods=["POST"])
def get_history():
    data = request.get_json()
    session_id = data.get("session_id")
    history = session_history.get(session_id, [])
    return jsonify({"session_id": session_id, "history": history}), 200 if history else 404
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7866)