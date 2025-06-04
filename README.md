# GenerativeAI_Text_To_Video

This project is a text-to-video generation pipeline that transforms textual prompts into high-quality videos. It leverages Tencent's HunyuanVideo model for video generation and Real-ESRGAN for video upscaling.

# Overview
The pipeline converts textual descriptions into coherent videos, enhancing them for clarity and resolution.

# Model Architecture
## HunyuanVideo
HunyuanVideo is a 13B parameter diffusion transformer model designed to be competitive with closed-source video foundation models and enable wider community access. This model uses a “dual-stream to single-stream” architecture to separately process the video and text tokens first, before concatenating and feeding them to the transformer to fuse the multimodal information. A pretrained multimodal large language model (MLLM) is used as the encoder because it has better image-text alignment, better image detail description and reasoning, and it can be used as a zero-shot learner if system instructions are added to user prompts. Finally, HunyuanVideo uses a 3D causal variational autoencoder to more efficiently process video data at the original resolution and frame rate.
