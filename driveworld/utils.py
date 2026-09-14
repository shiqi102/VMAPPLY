"""通用工具函数."""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image


def load_config(config_path: str = "config/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_scenarios(config: dict) -> dict:
    """获取配置中的场景定义."""
    return config.get("scenarios", {})


def get_scenario_names(config: dict) -> list[str]:
    """获取配置中所有场景名称."""
    return list(get_scenarios(config).keys())


def get_scenario_config(config: dict, scenario: str) -> dict:
    """获取单个场景配置."""
    scenario_cfg = get_scenarios(config).get(scenario, {})
    if isinstance(scenario_cfg, str):
        return {"text_prompt": scenario_cfg}
    return scenario_cfg or {}


def get_scenario_prompt(config: dict, scenario: str) -> str | None:
    """获取场景的文生视频提示词."""
    scenario_cfg = get_scenario_config(config, scenario)
    return scenario_cfg.get("text_prompt") or scenario_cfg.get("prompt")


def get_scenario_keyframe(config: dict, scenario: str) -> str | None:
    """获取场景的默认关键帧路径（图生视频）."""
    return get_scenario_config(config, scenario).get("keyframe")


def get_generation_seed(config: dict) -> int | None:
    """获取视频生成的默认随机种子."""
    seed = config.get("generation", {}).get("seed")
    return seed if seed is not None else None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.float16)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_video(frames: list[np.ndarray], output_path: str, fps: int = 8):
    """将帧列表保存为视频."""
    import imageio

    ensure_dir(Path(output_path).parent)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def frames_to_numpy(video_frames) -> list[np.ndarray]:
    """将 PIL / tensor 帧列表转为 numpy 数组."""
    result = []
    for frame in video_frames:
        if isinstance(frame, Image.Image):
            result.append(np.array(frame))
        elif isinstance(frame, torch.Tensor):
            arr = frame.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            result.append((arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8))
        elif isinstance(frame, np.ndarray):
            result.append(frame)
    return result
