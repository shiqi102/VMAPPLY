"""光流估计."""

import cv2
import numpy as np


def align_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """将 mask 对齐到 (H, W)，SAM2 等模型可能输出低分辨率 mask."""
    target_h, target_w = shape
    mask = np.squeeze(mask)
    if mask.shape[:2] == (target_h, target_w):
        return mask > 0

    resized = cv2.resize(
        (mask > 0).astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


class OpticalFlowEstimator:
    """基于 Farneback 的光流估计."""

    def __init__(self, config: dict):
        self.cfg = config["motion"]["optical_flow"]

    def compute(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=self.cfg["pyr_scale"],
            levels=self.cfg["levels"],
            winsize=self.cfg["winsize"],
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow

    def compute_video(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        flows = []
        for i in range(len(frames) - 1):
            flows.append(self.compute(frames[i], frames[i + 1]))
        return flows

    def flow_magnitude(self, flow: np.ndarray) -> np.ndarray:
        return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    def object_velocity(
        self,
        flow: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, float]:
        """从 mask 区域估计目标平均速度 (像素/帧)."""
        mask = align_mask_to_shape(mask, flow.shape[:2])
        if not mask.any():
            return 0.0, 0.0

        masked_flow = flow[mask]
        vx = float(np.mean(masked_flow[:, 0]))
        vy = float(np.mean(masked_flow[:, 1]))
        return vx, vy
