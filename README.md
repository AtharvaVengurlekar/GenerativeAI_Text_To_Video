# GenerativeAI_Text_To_Video

This project is a text-to-video generation pipeline that transforms textual prompts into high-quality videos. It leverages Tencent's HunyuanVideo model for video generation and Real-ESRGAN for video upscaling.

# Overview
The pipeline converts textual descriptions into coherent videos, enhancing them for clarity and resolution.

# Model Architecture
## HunyuanVideo
HunyuanVideo is a 13B parameter diffusion transformer model designed to be competitive with closed-source video foundation models and enable wider community access. This model uses a “dual-stream to single-stream” architecture to separately process the video and text tokens first, before concatenating and feeding them to the transformer to fuse the multimodal information. A pretrained multimodal large language model (MLLM) is used as the encoder because it has better image-text alignment, better image detail description and reasoning, and it can be used as a zero-shot learner if system instructions are added to user prompts. Finally, HunyuanVideo uses a 3D causal variational autoencoder to more efficiently process video data at the original resolution and frame rate.

<img width="1415" alt="Image" src="https://github.com/user-attachments/assets/737762ed-d241-4f98-95a8-e9d0bf6d04e7" />

1. Text Prompt Input
A user provides a natural language prompt that describes the desired video scene.
_Example: "Sunny day, freedom and adventure, high motion. A man with a beard riding a motorcycle on the street."_

2. Text Embedding with Large Language Model (LLM)
The text prompt is encoded into a sequence of text tokens using a large multimodal language model (LLM). These tokens capture semantic meaning and temporal cues.

3. Video Compression (Training Phase Only)
During training, ground-truth videos are encoded into a compact latent space using a Hunyuan Causal 3D Variational Autoencoder (3DVAE) encoder. This compresses the high-dimensional video data into manageable latent representations.

4. Noise Addition (Diffusion Process)
Gaussian noise is added to the latent video representations. This is a standard process in diffusion models, where the goal is to learn how to reverse the noise and recover the clean latent.

5. Dual-Stream Transformer Input
Both noisy latent video tokens and encoded text tokens are fed into the HunyuanVideo Diffusion Backbone, a transformer-based model. The dual-stream attention mechanism allows the model to effectively fuse spatial-temporal video features with text semantics.

6. Latent Video Output (Denoising)
The diffusion model denoises the input tokens step-by-step, generating a refined latent video representation conditioned on the text prompt.

7. Final Video Generation
The denoised latent video is decoded back into high-resolution video frames using the Hunyuan Causal 3DVAE Decoder, completing the text-to-video generation pipeline.
