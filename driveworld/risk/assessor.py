"""场景风险评估."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .metrics import RiskMetrics
from driveworld.utils import ensure_dir


class RiskAssessor:
    """综合风险评估，生成风险分数、热力图和关键风险帧."""

    def __init__(self, config: dict):
        self.cfg = config["risk"]
        self.metrics = RiskMetrics(config)
        self.weights = self.cfg["weights"]

    def assess(
        self,
        frames: list[np.ndarray],
        parse_result: dict,
        trajectory_result: dict,
        output_dir: str | None = None,
    ) -> dict:
        """完整风险评估."""
        trajectories = trajectory_result["trajectories"]
        ego_motion = trajectory_result["ego_motion"]
        road_parsing = parse_result["road_parsing"]
        segmented = parse_result["segmented_objects"]

        frame_risks = []
        for i, frame in enumerate(frames):
            frame_risk = self._assess_frame(
                i, frame, trajectories, ego_motion,
                road_parsing[i] if i < len(road_parsing) else {},
                segmented[i] if i < len(segmented) else [],
            )
            frame_risks.append(frame_risk)

        overall = self._aggregate(frame_risks)
        key_frames = self._find_key_risk_frames(frame_risks)

        result = {
            "overall_risk_score": overall["risk_score"],
            "overall_level": overall["level"],
            "frame_risks": frame_risks,
            "key_risk_frames": key_frames,
            "metrics_summary": overall,
            "trajectory_conflicts": self._check_all_conflicts(trajectories),
        }

        if output_dir:
            self._save_visualizations(frames, frame_risks, key_frames, output_dir)
            self._save_report(result, output_dir)

        return result

    def _assess_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        trajectories: list[dict],
        ego_motion: dict,
        road_info: dict,
        objects: list[dict],
    ) -> dict:
        ego_vel = np.array(ego_motion.get("velocity", [0.0, 0.0]))
        ego_pos = np.array([frame.shape[1] / 2, frame.shape[0] * 0.8])

        ttc_scores = []
        pet_scores = []
        intrusion_scores = []
        occlusion_scores = []

        for traj in trajectories:
            if frame_idx >= len(traj["positions"]):
                continue

            obj_pos = traj["positions"][frame_idx]
            obj_vel = traj["velocities"][frame_idx] if frame_idx < len(traj["velocities"]) else np.zeros(2)

            ttc = self.metrics.time_to_collision(ego_pos, ego_vel, obj_pos, obj_vel)
            if ttc < self.cfg["ttc_threshold"]:
                ttc_scores.append(1.0 - ttc / self.cfg["ttc_threshold"])

            if "drivable_mask" in road_info:
                intrusion = self.metrics.lane_intrusion(
                    np.array([obj_pos]), road_info["drivable_mask"]
                )
                intrusion_scores.append(intrusion)

        for obj in objects:
            occ = self.metrics.occlusion_score(
                obj["bbox"], objects, frame.shape[:2]
            )
            occlusion_scores.append(occ)

        visibility = self.metrics.visibility_degradation(frame)

        risk_score = (
            self.weights["ttc"] * (np.max(ttc_scores) if ttc_scores else 0.0)
            + self.weights["pet"] * (np.max(pet_scores) if pet_scores else 0.0)
            + self.weights["lane_intrusion"] * (np.max(intrusion_scores) if intrusion_scores else 0.0)
            + self.weights["occlusion"] * (np.max(occlusion_scores) if occlusion_scores else 0.0)
            + 0.1 * visibility
        )

        return {
            "frame_idx": frame_idx,
            "risk_score": float(np.clip(risk_score, 0.0, 1.0)),
            "ttc_risk": float(np.max(ttc_scores)) if ttc_scores else 0.0,
            "intrusion_risk": float(np.max(intrusion_scores)) if intrusion_scores else 0.0,
            "occlusion_risk": float(np.max(occlusion_scores)) if occlusion_scores else 0.0,
            "visibility_risk": visibility,
            "num_objects": len(objects),
        }

    def _aggregate(self, frame_risks: list[dict]) -> dict:
        scores = [f["risk_score"] for f in frame_risks]
        avg_score = float(np.mean(scores)) if scores else 0.0
        max_score = float(np.max(scores)) if scores else 0.0

        if max_score > 0.7:
            level = "high"
        elif max_score > 0.4:
            level = "medium"
        else:
            level = "low"

        return {
            "risk_score": avg_score,
            "max_risk_score": max_score,
            "level": level,
            "high_risk_frame_count": sum(1 for s in scores if s > 0.5),
        }

    def _find_key_risk_frames(
        self, frame_risks: list[dict], top_k: int = 5
    ) -> list[dict]:
        sorted_frames = sorted(frame_risks, key=lambda x: x["risk_score"], reverse=True)
        return sorted_frames[:top_k]

    def _check_all_conflicts(self, trajectories: list[dict]) -> list[dict]:
        conflicts = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                conflict = self.metrics.trajectory_conflict(trajectories[i], trajectories[j])
                if conflict["has_conflict"]:
                    conflicts.append({
                        "track_a": trajectories[i]["track_id"],
                        "track_b": trajectories[j]["track_id"],
                        "class_a": trajectories[i]["class_name"],
                        "class_b": trajectories[j]["class_name"],
                        **conflict,
                    })
        return conflicts

    def _save_visualizations(
        self,
        frames: list[np.ndarray],
        frame_risks: list[dict],
        key_frames: list[dict],
        output_dir: str,
    ):
        out = ensure_dir(Path(output_dir) / "risk_viz")

        # 风险时序曲线
        scores = [f["risk_score"] for f in frame_risks]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(scores, "r-", linewidth=2)
        ax.fill_between(range(len(scores)), scores, alpha=0.3, color="red")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Risk Score")
        ax.set_title("Scene Risk Timeline")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="orange", linestyle="--", label="Medium Risk")
        ax.axhline(y=0.7, color="red", linestyle="--", label="High Risk")
        ax.legend()
        fig.savefig(out / "risk_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 风险热力图
        heatmap = np.array(scores).reshape(1, -1)
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.imshow(heatmap, aspect="auto", cmap="hot", vmin=0, vmax=1)
        ax.set_xlabel("Frame")
        ax.set_title("Risk Heatmap")
        fig.savefig(out / "risk_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 关键风险帧标注
        for kf in key_frames[:3]:
            idx = kf["frame_idx"]
            if idx < len(frames):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(frames[idx])
                ax.set_title(
                    f"Key Risk Frame #{idx} | Score: {kf['risk_score']:.3f}"
                )
                ax.axis("off")
                fig.savefig(out / f"key_frame_{idx}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    def _save_report(self, result: dict, output_dir: str):
        out = ensure_dir(Path(output_dir))
        report = {
            "overall_risk_score": result["overall_risk_score"],
            "overall_level": result["overall_level"],
            "metrics_summary": result["metrics_summary"],
            "key_risk_frames": [
                {"frame_idx": kf["frame_idx"], "risk_score": kf["risk_score"]}
                for kf in result["key_risk_frames"]
            ],
            "trajectory_conflicts": result["trajectory_conflicts"],
        }
        with open(out / "risk_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
