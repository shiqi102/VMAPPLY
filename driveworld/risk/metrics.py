"""风险评估指标计算."""

import numpy as np


class RiskMetrics:
    """TTC、PET、轨迹冲突、车道侵入、遮挡等指标."""

    def __init__(self, config: dict):
        self.cfg = config["risk"]
        self.pixel_to_meter = 0.05  # 近似比例，实际应基于相机标定

    def time_to_collision(
        self,
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        obj_pos: np.ndarray,
        obj_vel: np.ndarray,
    ) -> float:
        """计算 TTC (Time To Collision)."""
        rel_pos = obj_pos - ego_pos
        rel_vel = obj_vel - ego_vel

        dist = np.linalg.norm(rel_pos) * self.pixel_to_meter
        rel_speed = np.linalg.norm(rel_vel) * self.pixel_to_meter

        if rel_speed < 0.01:
            return float("inf")

        # 仅当目标接近时计算
        closing = -np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) * np.linalg.norm(rel_vel) + 1e-8)
        if closing <= 0:
            return float("inf")

        return dist / rel_speed

    def post_encroachment_time(
        self,
        traj_a: dict,
        traj_b: dict,
    ) -> float:
        """计算 PET (Post-Encroachment Time)."""
        pos_a = traj_a["positions"]
        pos_b = traj_b["positions"]
        ts_a = traj_a["timestamps"]
        ts_b = traj_b["timestamps"]

        min_pet = float("inf")
        threshold = 30  # 像素

        for i, pa in enumerate(pos_a):
            for j, pb in enumerate(pos_b):
                dist = np.linalg.norm(pa - pb)
                if dist < threshold:
                    pet = abs(ts_a[i] - ts_b[j])
                    min_pet = min(min_pet, pet)

        return min_pet

    def trajectory_conflict(
        self,
        traj_a: dict,
        traj_b: dict,
        conflict_radius: float = 40.0,
    ) -> dict:
        """检测两条轨迹是否存在冲突."""
        pos_a = traj_a["positions"]
        pos_b = traj_b["positions"]

        min_dist = float("inf")
        conflict_frame = -1

        min_len = min(len(pos_a), len(pos_b))
        for i in range(min_len):
            dist = np.linalg.norm(pos_a[i] - pos_b[i])
            if dist < min_dist:
                min_dist = dist
                conflict_frame = i

        has_conflict = bool(min_dist < conflict_radius)
        return {
            "has_conflict": has_conflict,
            "min_distance": float(min_dist),
            "conflict_frame": int(conflict_frame),
            "severity": float(
                max(0.0, 1.0 - min_dist / conflict_radius) if has_conflict else 0.0
            ),
        }

    def lane_intrusion(
        self,
        obj_positions: np.ndarray,
        drivable_mask: np.ndarray,
    ) -> float:
        """计算目标侵入非可行驶区域的比例."""
        if len(obj_positions) == 0:
            return 0.0

        intrusions = 0
        h, w = drivable_mask.shape

        for pos in obj_positions:
            x, y = int(pos[0]), int(pos[1])
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            if drivable_mask[y, x] == 0:
                intrusions += 1

        return intrusions / len(obj_positions)

    def occlusion_score(
        self,
        obj_bbox: list,
        all_objects: list[dict],
        frame_shape: tuple,
    ) -> float:
        """估计目标被遮挡程度."""
        obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
        if obj_area == 0:
            return 0.0

        occluded_area = 0
        for other in all_objects:
            other_bbox = other.get("bbox", [0, 0, 0, 0])
            if other_bbox == obj_bbox:
                continue

            inter = self._bbox_intersection(obj_bbox, other_bbox)
            if inter > 0:
                # 仅统计位于前景 (更靠近图像底部) 的遮挡
                if other_bbox[3] > obj_bbox[3]:
                    occluded_area += inter

        return min(1.0, occluded_area / obj_area)

    def visibility_degradation(
        self,
        frame: np.ndarray,
        weather_type: str = "normal",
    ) -> float:
        """估计可见性退化程度."""
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame

        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))

        # 低对比度或极端亮度 -> 可见性退化
        contrast_score = max(0.0, 1.0 - contrast / 60.0)
        brightness_score = 0.0
        if brightness < 40:
            brightness_score = (40 - brightness) / 40.0
        elif brightness > 200:
            brightness_score = (brightness - 200) / 55.0

        weather_factor = {
            "normal": 0.0,
            "rain": 0.3,
            "night": 0.5,
            "fog": 0.6,
        }.get(weather_type, 0.0)

        return min(1.0, 0.4 * contrast_score + 0.3 * brightness_score + 0.3 * weather_factor)

    @staticmethod
    def _bbox_intersection(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)
