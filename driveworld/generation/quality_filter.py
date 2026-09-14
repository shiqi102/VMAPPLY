"""多轮生成与质量筛选机制."""

import cv2
import numpy as np
from typing import Callable


class QualityFilter:
    """对生成视频进行质量评估与筛选."""

    def __init__(self, config: dict):
        self.cfg = config["generation"]["quality_filter"]

    def compute_sharpness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def compute_brightness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray))

    def compute_temporal_consistency(self, frames: list[np.ndarray]) -> float:
        """基于相邻帧光流幅度估计时序一致性."""
        if len(frames) < 2:
            return 1.0

        diffs = []
        for i in range(len(frames) - 1):
            prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            diff = np.mean(np.abs(curr.astype(float) - prev.astype(float)))
            diffs.append(diff)

        mean_diff = np.mean(diffs)
        # 差异越小一致性越高，归一化到 [0, 1]
        consistency = max(0.0, 1.0 - mean_diff / 50.0)
        return consistency

    def evaluate(self, frames: list[np.ndarray]) -> dict:
        """评估单段视频质量."""
        sharpness_scores = [self.compute_sharpness(f) for f in frames]
        brightness_scores = [self.compute_brightness(f) for f in frames]
        temporal = self.compute_temporal_consistency(frames)

        avg_sharpness = np.mean(sharpness_scores)
        avg_brightness = np.mean(brightness_scores)

        passed = (
            avg_sharpness >= self.cfg["min_sharpness"]
            and self.cfg["min_brightness"] <= avg_brightness <= self.cfg["max_brightness"]
            and temporal >= self.cfg["min_temporal_consistency"]
        )

        return {
            "passed": passed,
            "sharpness": avg_sharpness,
            "brightness": avg_brightness,
            "temporal_consistency": temporal,
            "score": avg_sharpness * 0.4 + temporal * 100 * 0.6,
        }

    def generate_with_filter(
        self,
        generator_fn: Callable,
        *args,
        **kwargs,
    ) -> tuple[list[np.ndarray], dict]:
        """多轮生成，返回质量最优结果."""
        best_frames = None
        best_metrics = None
        max_retries = self.cfg["max_retries"]

        for attempt in range(max_retries):
            frames = generator_fn(*args, **kwargs)
            metrics = self.evaluate(frames)

            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_frames = frames
                best_metrics = metrics

            if metrics["passed"]:
                best_metrics["attempt"] = attempt + 1
                return best_frames, best_metrics

        best_metrics["attempt"] = max_retries
        best_metrics["passed"] = False
        return best_frames, best_metrics
