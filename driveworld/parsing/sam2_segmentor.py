"""SAM2 视频目标分割."""

import cv2
import numpy as np
import torch
from PIL import Image

from driveworld.utils import get_device


class SAM2Segmentor:
    """基于 SAM2 的视频目标分割器."""

    def __init__(self, config: dict):
        self.cfg = config["parsing"]["sam2"]
        self.device = get_device()
        self._predictor = None

    def _load_model(self):
        if self._predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model_cfg = "configs/sam2/sam2_hiera_l.yaml"
            checkpoint = self._download_checkpoint()
            sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
            self._predictor = SAM2ImagePredictor(sam2_model)
        except ImportError:
            # 回退到 HuggingFace transformers 接口
            from transformers import Sam2Model, Sam2Processor

            self._hf_model = Sam2Model.from_pretrained(self.cfg["model_id"]).to(self.device)
            self._hf_processor = Sam2Processor.from_pretrained(self.cfg["model_id"])
            self._predictor = "hf"

    def _download_checkpoint(self) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=self.cfg["model_id"],
            filename="sam2_hiera_large.pt",
        )

    def segment_frame(
        self,
        frame: np.ndarray,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: list | None = None,
    ) -> dict:
        """对单帧进行分割."""
        self._load_model()

        if self._predictor == "hf":
            return self._segment_hf(frame, point_coords, point_labels, box)
        return self._segment_native(frame, point_coords, point_labels, box)

    def _segment_native(
        self,
        frame: np.ndarray,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: list | None,
    ) -> dict:
        self._predictor.set_image(frame)

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.array(box) if box else None,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        best_mask = self._resize_mask_to_frame(masks[best_idx], frame.shape[:2])
        return {
            "masks": masks,
            "best_mask": best_mask,
            "score": float(scores[best_idx]),
        }

    def _segment_hf(
        self,
        frame: np.ndarray,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: list | None,
    ) -> dict:
        image = Image.fromarray(frame)
        inputs = {"image": image}

        if point_coords is not None:
            inputs["input_points"] = [point_coords.tolist()]
            inputs["input_labels"] = [point_labels.tolist()]

        processed = self._hf_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._hf_model(**processed)

        masks = outputs.pred_masks.squeeze().cpu().numpy()
        scores = outputs.iou_scores.squeeze().cpu().numpy()

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx] if masks.ndim == 3 else masks
        best_mask = self._resize_mask_to_frame(best_mask, frame.shape[:2])
        return {
            "masks": masks,
            "best_mask": best_mask,
            "score": float(scores[best_idx]) if scores.ndim > 0 else float(scores),
        }

    def _resize_mask_to_frame(
        self, mask: np.ndarray, frame_shape: tuple[int, int]
    ) -> np.ndarray:
        """将分割 mask 还原到原图分辨率."""
        h, w = frame_shape
        mask = np.squeeze(mask)
        if mask.shape[:2] == (h, w):
            return mask.astype(np.uint8)
        return cv2.resize(
            (mask > 0).astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    def segment_objects(
        self,
        frame: np.ndarray,
        objects: list[dict],
    ) -> list[dict]:
        """基于检测框对多个目标进行精细分割."""
        results = []
        for obj in objects:
            bbox = obj["bbox"]
            box = [bbox[0], bbox[1], bbox[2], bbox[3]]
            seg_result = self.segment_frame(frame, box=box)
            results.append({
                **obj,
                "mask": seg_result["best_mask"],
                "seg_score": seg_result["score"],
            })
        return results

    def segment_video(
        self,
        frames: list[np.ndarray],
        objects_per_frame: list[list[dict]] | None = None,
    ) -> list[list[dict]]:
        """对视频逐帧分割."""
        all_results = []
        for i, frame in enumerate(frames):
            objs = objects_per_frame[i] if objects_per_frame else []
            if objs:
                all_results.append(self.segment_objects(frame, objs))
            else:
                all_results.append([])
        return all_results
