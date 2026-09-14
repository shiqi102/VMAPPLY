"""SegFormer 道路结构解析."""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from driveworld.utils import get_device


# Cityscapes 类别中道路相关标签
CITYSCAPES_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


class SegFormerRoadParser:
    """基于 SegFormer 的道路结构语义分割."""

    def __init__(self, config: dict):
        self.cfg = config["parsing"]["segformer"]
        self.device = get_device()
        self.road_classes = set(self.cfg["road_classes"])

        self.processor = SegformerImageProcessor.from_pretrained(self.cfg["model_id"])
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.cfg["model_id"]
        ).to(self.device)
        self.model.eval()

    def parse(self, frame: np.ndarray) -> dict:
        """对单帧进行道路结构解析."""
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        road_mask = np.isin(seg_map, list(self.road_classes)).astype(np.uint8)
        drivable_mask = (seg_map == 0).astype(np.uint8)

        # 提取车道线区域 (道路边缘)
        road_edges = self._extract_lane_boundaries(drivable_mask)

        # 动态目标类别
        dynamic_classes = {11, 12, 13, 14, 15, 16, 17, 18}
        dynamic_mask = np.isin(seg_map, list(dynamic_classes)).astype(np.uint8)

        detected_objects = self._extract_objects(seg_map, dynamic_classes)

        return {
            "seg_map": seg_map,
            "road_mask": road_mask,
            "drivable_mask": drivable_mask,
            "lane_boundaries": road_edges,
            "dynamic_mask": dynamic_mask,
            "objects": detected_objects,
        }

    def parse_video(self, frames: list[np.ndarray]) -> list[dict]:
        return [self.parse(f) for f in frames]

    def _extract_lane_boundaries(self, drivable_mask: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(drivable_mask * 255, 50, 150)
        return edges

    def _extract_objects(
        self, seg_map: np.ndarray, dynamic_classes: set
    ) -> list[dict]:
        objects = []
        for cls_id in dynamic_classes:
            mask = (seg_map == cls_id).astype(np.uint8)
            if mask.sum() == 0:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                objects.append({
                    "class_id": cls_id,
                    "class_name": CITYSCAPES_CLASSES.get(cls_id, "unknown"),
                    "bbox": [x, y, x + w, y + h],
                    "area": area,
                    "centroid": [x + w // 2, y + h // 2],
                })
        return objects
