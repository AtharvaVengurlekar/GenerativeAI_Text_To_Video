# GenerativeAI_Text_To_Video

This project is a text-to-video generation pipeline that transforms textual prompts into high-quality videos. It leverages Tencent's HunyuanVideo model for video generation and Real-ESRGAN for video upscaling.

# Overview
The pipeline converts textual descriptions into coherent videos, enhancing them for clarity and resolution.

# Model Architecture
## HunyuanVideo Tencent
HunyuanVideo is a 13B parameter diffusion transformer model designed to be competitive with closed-source video foundation models and enable wider community access. This model uses a “dual-stream to single-stream” architecture to separately process the video and text tokens first, before concatenating and feeding them to the transformer to fuse the multimodal information. A pretrained multimodal large language model (MLLM) is used as the encoder because it has better image-text alignment, better image detail description and reasoning, and it can be used as a zero-shot learner if system instructions are added to user prompts. Finally, HunyuanVideo uses a 3D causal variational autoencoder to more efficiently process video data at the original resolution and frame rate.

<img width="1415" alt="Image" src="https://github.com/user-attachments/assets/737762ed-d241-4f98-95a8-e9d0bf6d04e7" />

1. Text Prompt Input
Users provide a natural language prompt (e.g., "Sunny day, adventure, man on a motorcycle").

2. Text Embedding with Large Language Model (LLM)
A large multimodal language model (LLM) encodes the prompt into text tokens capturing semantic and temporal information.

3. Video Encoding (Training Only)
Ground-truth videos are compressed into latent representations using the Hunyuan Causal 3D VAE Encoder.

4. Diffusion (Noise Injection)
Gaussian noise is added to video latents as part of the diffusion learning process.

5.  Fusion via Transformer
Noisy video tokens and text tokens are fused using a dual-stream to single-stream transformer (HunyuanVideo Backbone).

6. Denoising
The model denoises the latent video step-by-step, guided by the input prompt.

7.  Final Video Generation
The Hunyuan 3DVAE Decoder reconstructs high-quality video frames from the denoised latents.

## ESRGAN : Enhanced Super-Resolution Generative Adversarial Networks
ESRGAN is designed to convert low-resolution (LR) images into high-resolution (HR) images with photorealistic detail.

## model network architecture 
![Image](https://github.com/user-attachments/assets/937983ff-c76a-40b3-b0ea-5272251420a7)

### 1. Generator Network
  - Takes a low-resolution image as input.

  - Uses a deep ResNet-based architecture built with Residual-in-Residual Dense Blocks (RRDBs) for efficient feature extraction.

  - Each RRDB uses:

    - Dense connections (for feature reuse)

    - Residual connections (for stability)

    - No Batch Normalization (to preserve range)

### 2. Upsampling
  - Pixel-shuffle layers upscale the image (e.g., 2x or 4x).

### 3.  Discriminator
  - A PatchGAN-style discriminator analyzes image patches to distinguish real vs. generated images.

  - Encourages sharper, more realistic textures.

### 4. Perceptual Loss (VGG Feature Loss)
  - Utilizes VGG-based feature loss instead of just pixel-wise errors.

  - Enhances texture quality and perceptual similarity to the ground truth.

### 5. Training Objective (Adversarial + Content + Perceptual Loss)

  - Combines content loss (MSE/L1), perceptual loss, and adversarial loss for optimal results.
 
### 6. Output

  - Produces high-resolution, photorealistic images from low-resolution inputs — ideal for video enhancement and image restoration.

# Hardware Requirements

To run the full pipeline, the following hardware specifications are recommended:

| Model         | Resolution (H × W × Frames) | GPU Peak Memory |
|---------------|-----------------------------|------------------|
| HunyuanVideo  | 720 × 1280 × 129            |     60 GB        |
| HunyuanVideo  | 544 × 960 × 129             |     45 GB        |

Note: A minimum of NVIDIA RTX 6000 GPU with 50GB memory is recommended.

# Sample Prompt and Output
Prompt: "Create a 6-second cinematic video clip featuring a stylish human figure. The person should be wearing modern, luxurious sunglasses (reflective lenses, metallic frames) and exhibit an extravagant, fashionable dress sense: think designer clothes with bold patterns, layered textures, and accessorized with jewelry like rings, bracelets, and a sleek watch. The clothing should blend streetwear and high-fashion styles, like a bright patterned blazer, tailored pants, and exclusive sneakers or leather shoes. The human should have a confident walking posture or a slow-motion head turn, showing off the sunglasses prominently. The scene should have excellent lighting that highlights clothing details and lens reflections, with a high-definition urban or upscale city background. Focus heavily on realistic skin textures, fabric movement, light glinting off accessories, and natural hair flow (if visible). Overall tone: chic, modern, luxurious."

Generated Video:
https://github.com/user-attachments/assets/d1c94a1a-2861-45d2-a977-f432470e6296

prompt: "Create a short video in the style of a fresh, USA-style film photo. The scene features a young American woman standing casually on a quiet urban street during golden hour. She is holding a coffee cup in one hand and wearing a textured green crossbody bag from Victoria's Secret. Her outfit includes a loose grey sweater, and her natural, flowing hair catches the soft light. She has a gentle smile and her eyes look sideways at the camera, conveying a peaceful, candid mood. The background shows softly blurred buildings and street elements, bathed in the warm, natural sunlight of a setting sun. The composition should be a mid-range, half-length portrait with soft, cinematic light and subtle shadows that enhance the calm, serene atmosphere. The camera angle should feel natural and slightly handheld, evoking a film-like, authentic moment."

Generated Video: 
https://github.com/user-attachments/assets/87f54fd1-c217-49c7-94a0-6b82243306da

# Setup Instructions

Step 1: Clone the Repository:
```bash
git clone https://github.com/AtharvaVengurlekar/GenerativeAI_Text_To_Video.git
cd GenerativeAI_Text_To_Video
```
Step 2: Create a Conda Environment:
```bash
conda create -n text2video python=3.10.6
conda activate text2video
```
Step 3: Install Dependencies:
```bash
pip install -r requirements.txt
```
Step 4 : Download Model Files:
Download it from the following link and place it in the root directory of the project.
https://drive.google.com/drive/folders/17lx2X-I6f9V088VdTuys_Vzz7N9ipWLN?usp=sharing

OR Run this script:
```bash
download.py
```

# Testing the Pipeline

1️) Basic HunyuanVideo Test (Text ➝ Video Only)
To test the basic functionality of the HunyuanVideo model (without Real-ESRGAN upscaling), run:

```bash
python test.py
```

- Input: A text prompt

- Output: A generated video saved in the main directory

- Real-ESRGAN is not used in this test

- Use this script to validate the core text-to-video pipeline


2) Full Pipeline Test with ESRGAN (Text ➝ High-Res Video)
To generate high-resolution 4K videos, including Real-ESRGAN upscaling, run:

```bash
python finalpipeline.py
```

- Generate video using HunyuanVideo

- Enhance and upscale it using Real-ESRGAN

- Output: A high-quality upscaled video saved in the main directory

Make sure RealESRGAN_x4plus.pth is placed in the root directory

3) API Access via Postman (Run Flask App)
To expose the model as an API and test it using Postman or any other API client:

```bash
python app.py
```

##  Environment Variables & Credentials Setup
To run the Flask API successfully (app.py), you need to configure the following credentials in a .env file in your project root.

 Create a .env file

```env
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
REGION_NAME=your-region-name
BUCKET_NAME=your-s3-bucket-name
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
CUDA_DEVICE_ID=0
```

## How to Get These Credentials
- AWS_ACCESS_KEY & AWS_SECRET_KEY : 
Go to the AWS IAM Console, create a user with programmatic access, and assign S3 full access permissions. Copy the access and secret keys.

- BUCKET_NAME : 
Go to the AWS S3 Console, create a new bucket, and use its name here.

- REGION_NAME : 
Use the region where your S3 bucket is located (e.g., ap-south-1 for Mumbai).

- MONGODB_URI : 
Create a free cluster on MongoDB Atlas, then click Connect > Drivers > Copy Connection String and paste it here.

# 📡 API Usage via cURL
Once the server is running, you can interact with it using the following cURL commands:

1) Generate a Video
```bash
curl --location 'https://your-domain.com/generate_video' \
--header 'Content-Type: application/json' \
--data '{
  "prompt": "A man sitting at a coffee shop, drinking coffee",
  "width": 960,
  "height": 544,
  "session_id": "15"
}'
```
2) Get Status of a Request
```bash
curl --location 'https://your-domain.com/get_status' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "15",
  "reference_id": "your_reference_id"
}'
```
3) Download the Generated Video
```bash
curl --location 'https://your-domain.com/download_video/your_reference_id' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "15"
}'
```
4) View Generation History
```bash
curl --location 'https://your-domain.com/history' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "15"
}'
```
Replace https://your-domain.com with your actual hosted URL.

##  How to Host the API
To make your API publicly accessible, follow these steps:

1) Use a Cloud VM Provider
 Host the Flask app on services like:

 - AWS EC2

 - DigitalOcean Droplet

 - Render.com

 - Railway.app

2) Expose Your Flask App
 Use gunicorn + nginx or platforms like:

 - Railway

 - Render

3) Enable HTTPS
 Use Let's Encrypt with Nginx or enable auto-SSL if using Render/Railway.
