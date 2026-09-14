"""Stable Video Diffusion 关键帧到视频生成."""

from pathlib import Path

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

from driveworld.utils import get_device, get_dtype, ensure_dir, frames_to_numpy


class SVDGenerator:
    """基于 HuggingFace Stable Video Diffusion 的关键帧到视频生成器."""

    def __init__(self, config: dict):
        self.cfg = config["generation"]["svd"]
        self.device = get_device()
        self.dtype = get_dtype(self.cfg.get("dtype", "float16"))

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg["model_id"],
            torch_dtype=self.dtype,
            variant="fp16",
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        keyframe: str | Image.Image,
        seed: int | None = None,
        output_path: str | None = None,
    ) -> list:
        """关键帧到视频生成."""
        if isinstance(keyframe, str):
            image = load_image(keyframe)
        else:
            image = keyframe

        image = image.resize((1024, 576))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            image,
            decode_chunk_size=self.cfg["decode_chunk_size"],
            num_frames=self.cfg["num_frames"],
            motion_bucket_id=self.cfg["motion_bucket_id"],
            noise_aug_strength=self.cfg["noise_aug_strength"],
            num_inference_steps=self.cfg["num_inference_steps"],
            generator=generator,
        )

        frames = result.frames[0]

        if output_path:
            ensure_dir(Path(output_path).parent)
            export_to_video(frames, output_path, fps=self.cfg["fps"])

        return frames_to_numpy(frames)
