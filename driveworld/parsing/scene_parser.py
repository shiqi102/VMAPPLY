"""场景解析统一接口."""

import numpy as np

from .segformer_road import SegFormerRoadParser
from .sam2_segmentor import SAM2Segmentor


class SceneParser:
    """整合 SegFormer 道路解析与 SAM2 目标分割."""

    def __init__(self, config: dict):
        self.road_parser = SegFormerRoadParser(config)
        self.segmentor = SAM2Segmentor(config)

    def parse(self, frames: list[np.ndarray]) -> dict:
        """完整场景解析."""
        road_results = self.road_parser.parse_video(frames)

        segmented_objects = []
        for i, frame in enumerate(frames):
            objects = road_results[i]["objects"]
            if objects:
                seg_objs = self.segmentor.segment_objects(frame, objects)
            else:
                seg_objs = []
            segmented_objects.append(seg_objs)

        return {
            "num_frames": len(frames),
            "road_parsing": road_results,
            "segmented_objects": segmented_objects,
            "summary": self._build_summary(road_results, segmented_objects),
        }

    def _build_summary(
        self,
        road_results: list[dict],
        segmented_objects: list[list[dict]],
    ) -> dict:
        all_classes = set()
        total_objects = 0
        for frame_objs in segmented_objects:
            total_objects += len(frame_objs)
            for obj in frame_objs:
                all_classes.add(obj.get("class_name", "unknown"))

        road_coverage = [
            r["drivable_mask"].sum() / max(r["drivable_mask"].size, 1)
            for r in road_results
        ]

        return {
            "total_objects_detected": total_objects,
            "object_classes": list(all_classes),
            "avg_road_coverage": float(np.mean(road_coverage)),
            "frames_with_pedestrians": sum(
                1 for objs in segmented_objects
                if any(o.get("class_name") in ("person", "rider") for o in objs)
            ),
            "frames_with_vehicles": sum(
                1 for objs in segmented_objects
                if any(o.get("class_name") in ("car", "truck", "bus", "motorcycle") for o in objs)
            ),
        }
