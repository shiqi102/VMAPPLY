"""目标轨迹提取与运动建模."""

import numpy as np

from .optical_flow import OpticalFlowEstimator, align_mask_to_shape


class TrajectoryExtractor:
    """从连续帧 bbox/mask/光流信息提取交通参与者轨迹."""

    def __init__(self, config: dict):
        self.cfg = config["motion"]["tracking"]
        self.flow_estimator = OpticalFlowEstimator(config)

    def extract(
        self,
        frames: list[np.ndarray],
        segmented_objects: list[list[dict]],
        fps: float = 8.0,
    ) -> dict:
        """提取所有目标的轨迹."""
        flows = self.flow_estimator.compute_video(frames)
        tracks = self._build_tracks(segmented_objects)
        trajectories = self._compute_trajectories(tracks, flows, fps)

        ego_motion = self._estimate_ego_motion(flows, segmented_objects)

        return {
            "trajectories": trajectories,
            "tracks": tracks,
            "ego_motion": ego_motion,
            "num_tracks": len(trajectories),
        }

    def _build_tracks(
        self, segmented_objects: list[list[dict]]
    ) -> dict[int, list[dict]]:
        """基于 IoU 的简单多目标跟踪."""
        tracks: dict[int, list[dict]] = {}
        next_id = 0
        active_tracks: dict[int, dict] = {}

        for frame_idx, frame_objs in enumerate(segmented_objects):
            matched = set()

            for obj in frame_objs:
                best_track_id = None
                best_iou = 0.0

                for tid, last_obj in active_tracks.items():
                    if tid in matched:
                        continue
                    iou = self._compute_iou(obj["bbox"], last_obj["bbox"])
                    if iou > self.cfg["iou_threshold"] and iou > best_iou:
                        best_iou = iou
                        best_track_id = tid

                if best_track_id is not None:
                    matched.add(best_track_id)
                    entry = {**obj, "frame_idx": frame_idx}
                    tracks[best_track_id].append(entry)
                    active_tracks[best_track_id] = obj
                else:
                    entry = {**obj, "frame_idx": frame_idx}
                    tracks[next_id] = [entry]
                    active_tracks[next_id] = obj
                    matched.add(next_id)
                    next_id += 1

            # 清理长时间未匹配的轨迹
            for tid in list(active_tracks.keys()):
                if tid not in matched:
                    del active_tracks[tid]

        # 过滤短轨迹
        min_len = self.cfg["min_track_length"]
        return {tid: traj for tid, traj in tracks.items() if len(traj) >= min_len}

    def _compute_trajectories(
        self,
        tracks: dict[int, list[dict]],
        flows: list[np.ndarray],
        fps: float,
    ) -> list[dict]:
        trajectories = []

        for track_id, track in tracks.items():
            positions = []
            velocities = []
            timestamps = []

            for entry in track:
                cx, cy = entry["centroid"]
                frame_idx = entry["frame_idx"]
                positions.append([cx, cy])
                timestamps.append(frame_idx / fps)

                if frame_idx < len(flows) and "mask" in entry:
                    mask = entry["mask"]
                    if isinstance(mask, np.ndarray):
                        vx, vy = self.flow_estimator.object_velocity(
                            flows[frame_idx], mask.astype(np.uint8)
                        )
                        velocities.append([vx * fps, vy * fps])
                    else:
                        velocities.append([0.0, 0.0])
                else:
                    velocities.append([0.0, 0.0])

            speed = np.linalg.norm(velocities, axis=1) if velocities else [0.0]
            class_name = track[0].get("class_name", "unknown")

            trajectories.append({
                "track_id": track_id,
                "class_name": class_name,
                "positions": np.array(positions),
                "velocities": np.array(velocities),
                "timestamps": np.array(timestamps),
                "mean_speed": float(np.mean(speed)),
                "max_speed": float(np.max(speed)),
                "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
            })

        return trajectories

    def _estimate_ego_motion(
        self,
        flows: list[np.ndarray],
        segmented_objects: list[list[dict]],
    ) -> dict:
        """估计自车运动 (排除动态目标区域后的背景光流)."""
        ego_velocities = []

        for i, flow in enumerate(flows):
            dynamic_mask = np.zeros(flow.shape[:2], dtype=bool)
            if i < len(segmented_objects):
                for obj in segmented_objects[i]:
                    if "mask" in obj and isinstance(obj["mask"], np.ndarray):
                        dynamic_mask |= align_mask_to_shape(
                            obj["mask"], flow.shape[:2]
                        )

            static_mask = ~dynamic_mask
            if static_mask.sum() > 0:
                static_flow = flow[static_mask]
                ego_vx = float(np.median(static_flow[:, 0]))
                ego_vy = float(np.median(static_flow[:, 1]))
                ego_velocities.append([ego_vx, ego_vy])

        if not ego_velocities:
            return {"velocity": [0.0, 0.0], "speed": 0.0}

        avg_vel = np.mean(ego_velocities, axis=0)
        return {
            "velocity": avg_vel.tolist(),
            "speed": float(np.linalg.norm(avg_vel)),
        }

    @staticmethod
    def _compute_iou(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
