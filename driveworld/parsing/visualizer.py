"""SegFormer / SAM2 / 跟踪结果可视化."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from driveworld.parsing.segformer_road import CITYSCAPES_CLASSES
from driveworld.utils import ensure_dir, save_video


# Cityscapes 19 类配色 (R, G, B)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # 0 road
    [244, 35, 232],   # 1 sidewalk
    [70, 70, 70],     # 2 building
    [102, 102, 156],  # 3 wall
    [190, 153, 153],  # 4 fence
    [153, 153, 153],  # 5 pole
    [250, 170, 30],   # 6 traffic light
    [220, 220, 0],    # 7 traffic sign
    [107, 142, 35],   # 8 vegetation
    [152, 251, 152],  # 9 terrain
    [70, 130, 180],   # 10 sky
    [220, 20, 60],    # 11 person
    [255, 0, 0],      # 12 rider
    [0, 0, 142],      # 13 car
    [0, 0, 70],       # 14 truck
    [0, 60, 100],     # 15 bus
    [0, 80, 100],     # 16 train
    [0, 0, 230],      # 17 motorcycle
    [119, 11, 32],    # 18 bicycle
], dtype=np.uint8)

TRACK_COLORS = [
    (255, 128, 0), (0, 255, 128), (128, 0, 255), (0, 200, 255),
    (255, 0, 128), (200, 255, 0), (255, 200, 0), (0, 128, 255),
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 0),
]


class ParsingVisualizer:
    """保存道路分割、目标 mask 与 IoU 跟踪可视化."""

    def __init__(self, config: dict):
        viz_cfg = config.get("output", {}).get("parsing_viz", {})
        self.max_frames = int(viz_cfg.get("max_frames", 10))
        self.save_video = bool(viz_cfg.get("save_video", True))
        self.alpha = float(viz_cfg.get("overlay_alpha", 0.45))

    def save_all(
        self,
        frames: list[np.ndarray],
        parse_result: dict,
        tracks: dict[int, list[dict]],
        output_dir: str | Path,
    ) -> Path:
        out = ensure_dir(Path(output_dir) / "parsing_viz")
        seg_dir = ensure_dir(out / "segformer")
        sam_dir = ensure_dir(out / "sam2")
        track_dir = ensure_dir(out / "tracking")

        self._save_class_legend(out / "legend_classes.png")

        frame_indices = self._sample_indices(len(frames))
        overview_frames = []

        for idx in frame_indices:
            frame = frames[idx]
            road = parse_result["road_parsing"][idx]
            objects = parse_result["segmented_objects"][idx]

            seg_panel = self.render_segformer(frame, road)
            sam_panel = self.render_sam2(frame, objects)
            track_panel = self.render_tracking(
                frame, idx, tracks, parse_result["segmented_objects"]
            )

            cv2.imwrite(str(seg_dir / f"frame_{idx:03d}.png"), seg_panel)
            cv2.imwrite(str(sam_dir / f"frame_{idx:03d}.png"), sam_panel)
            cv2.imwrite(str(track_dir / f"frame_{idx:03d}.png"), track_panel)

            overview = self._compose_overview(seg_panel, sam_panel, track_panel)
            cv2.imwrite(str(out / f"overview_{idx:03d}.png"), overview)
            overview_frames.append(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB))

        if self.save_video and overview_frames:
            save_video(overview_frames, str(out / "parsing_overview.mp4"), fps=4)

        return out

    def _sample_indices(self, num_frames: int) -> list[int]:
        if num_frames <= self.max_frames:
            return list(range(num_frames))
        return [
            int(round(i * (num_frames - 1) / (self.max_frames - 1)))
            for i in range(self.max_frames)
        ]

    def render_segformer(self, frame: np.ndarray, road: dict) -> np.ndarray:
        """SegFormer: 语义分割 + 可行驶区域 + 道路边缘(车道线近似)."""
        h, w = frame.shape[:2]
        seg_map = road["seg_map"]
        base = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 左: 全图语义着色
        semantic = CITYSCAPES_PALETTE[np.clip(seg_map, 0, 18)].astype(np.uint8)
        semantic_bgr = cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR)

        # 右: 原图 + 可行驶区域 + 道路边缘
        overlay = base.copy()
        drivable = road["drivable_mask"] > 0
        road_sidewalk = road["road_mask"] > 0

        green = np.zeros_like(overlay)
        green[drivable] = (0, 200, 0)
        overlay = cv2.addWeighted(overlay, 1.0, green, self.alpha, 0)

        purple = np.zeros_like(overlay)
        sidewalk_only = road_sidewalk & ~drivable
        purple[sidewalk_only] = (200, 0, 200)
        overlay = cv2.addWeighted(overlay, 1.0, purple, self.alpha * 0.6, 0)

        lane_edges = road["lane_boundaries"] > 0
        overlay[lane_edges] = (0, 255, 255)

        # 动态目标粗检框 (SegFormer bbox，SAM2 之前)
        for obj in road.get("objects", []):
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)
            cv2.putText(
                overlay, f"SF:{obj['class_name']}", (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA,
            )

        self._draw_title(semantic_bgr, "19-class semantic")
        self._draw_title(overlay, "drivable(green) sidewalk(purple) edges(cyan)")

        panel = np.zeros((h, w * 2, 3), dtype=np.uint8)
        panel[:, :w] = semantic_bgr
        panel[:, w:] = overlay
        return panel

    def render_sam2(self, frame: np.ndarray, objects: list[dict]) -> np.ndarray:
        """SAM2: 基于 SegFormer bbox 的精细 mask."""
        base = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

        for i, obj in enumerate(objects):
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
            mask = obj.get("mask")
            if isinstance(mask, np.ndarray):
                mask_bool = mask.squeeze() > 0
                if mask_bool.shape[:2] != base.shape[:2]:
                    mask_bool = cv2.resize(
                        mask_bool.astype(np.uint8),
                        (base.shape[1], base.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                colored = np.zeros_like(base)
                colored[mask_bool] = color
                base = cv2.addWeighted(base, 1.0, colored, self.alpha, 0)
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(base, contours, -1, color, 2)

            x1, y1, x2, y2 = obj["bbox"]
            label = obj.get("class_name", "obj")
            score = obj.get("seg_score")
            text = f"SAM2:{label}"
            if score is not None:
                text += f" {score:.2f}"
            cv2.rectangle(base, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                base, text, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        self._draw_title(base, "SAM2: refined masks (SegFormer bbox -> SAM2 segment)")
        if not objects:
            cv2.putText(
                base, "No objects in this frame", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return base

    def render_tracking(
        self,
        frame: np.ndarray,
        frame_idx: int,
        tracks: dict[int, list[dict]],
        segmented_objects: list[list[dict]],
    ) -> np.ndarray:
        """IoU 多目标跟踪: 非 SAM2 内置，由轨迹模块基于 bbox 关联."""
        base = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()

        # 当前帧各 track 的检测
        active = []
        for track_id, entries in tracks.items():
            for entry in entries:
                if entry["frame_idx"] == frame_idx:
                    active.append((track_id, entry))

        for track_id, entry in active:
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            x1, y1, x2, y2 = entry["bbox"]
            cx, cy = entry["centroid"]

            # 历史轨迹线
            history = [
                e["centroid"] for e in tracks[track_id]
                if e["frame_idx"] <= frame_idx
            ]
            for j in range(1, len(history)):
                p1 = tuple(int(v) for v in history[j - 1])
                p2 = tuple(int(v) for v in history[j])
                cv2.line(base, p1, p2, color, 2, cv2.LINE_AA)

            cv2.rectangle(base, (x1, y1), (x2, y2), color, 2)
            cv2.circle(base, (int(cx), int(cy)), 4, color, -1)
            cv2.putText(
                base,
                f"ID:{track_id} {entry.get('class_name', '')}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        self._draw_title(
            base,
            f"Tracking (IoU): {len(active)} active tracks | total tracks: {len(tracks)}",
        )
        return base

    def _compose_overview(
        self, seg_panel: np.ndarray, sam_panel: np.ndarray, track_panel: np.ndarray
    ) -> np.ndarray:
        target_w = seg_panel.shape[1]
        sam = self._resize_width(sam_panel, target_w)
        track = self._resize_width(track_panel, target_w)
        return np.vstack([seg_panel, sam, track])

    @staticmethod
    def _resize_width(img: np.ndarray, width: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w == width:
            return img
        scale = width / w
        return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def _save_class_legend(self, path: Path):
        rows = len(CITYSCAPES_CLASSES)
        cell_h, cell_w = 28, 280
        legend = np.ones((rows * cell_h + 40, cell_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            legend, "Cityscapes classes (SegFormer)", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
        )
        for cls_id, name in CITYSCAPES_CLASSES.items():
            y = 40 + cls_id * cell_h
            color = tuple(int(c) for c in CITYSCAPES_PALETTE[cls_id][::-1])
            cv2.rectangle(legend, (10, y + 4), (40, y + cell_h - 4), color, -1)
            note = name
            if cls_id == 0:
                note += " (drivable)"
            if cls_id in {11, 12, 13, 14, 15, 16, 17, 18}:
                note += " -> SAM2 target"
            cv2.putText(
                legend, f"{cls_id}: {note}", (50, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(path), legend)

    @staticmethod
    def _draw_title(img: np.ndarray, text: str, y: int = 28, x: int = 10):
        cv2.rectangle(img, (x, y - 22), (min(x + len(text) * 9 + 10, img.shape[1] - 2), y + 6), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
