"""CogVideoX-5B 文本到视频生成."""

from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from driveworld.utils import get_device, get_dtype, ensure_dir, frames_to_numpy


class CogVideoXGenerator:
    """基于 HuggingFace CogVideoX-5B 的文本到视频生成器."""

    def __init__(self, config: dict):
        self.cfg = config["generation"]["cogvideox"]
        self.device = get_device()
        self.dtype = get_dtype(self.cfg.get("dtype", "bfloat16"))

        self.pipe = CogVideoXPipeline.from_pretrained(
            self.cfg["model_id"],
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int | None = None,
        output_path: str | None = None,
    ) -> list:
        """文本到视频生成."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_videos_per_prompt=1,
            num_inference_steps=self.cfg["num_inference_steps"],
            num_frames=self.cfg["num_frames"],
            guidance_scale=self.cfg["guidance_scale"],
            generator=generator,
        )

        frames = result.frames[0]

        if output_path:
            ensure_dir(Path(output_path).parent)
            export_to_video(frames, output_path, fps=self.cfg["fps"])

        return frames_to_numpy(frames)
