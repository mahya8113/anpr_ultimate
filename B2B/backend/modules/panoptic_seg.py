"""
panoptic_seg.py - قطعه‌بندی Panoptic (ترکیب semantic و instance segmentation)
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PanopticSegmenter:
    def init(self, semantic_model_name: str = "nvidia/mit-b0", instance_model_path: str = "yolov8n-seg.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.semantic_model = None
        self.instance_model = None
        try:
            from transformers import AutoModelForSemanticSegmentation
            self.semantic_model = AutoModelForSemanticSegmentation.from_pretrained(semantic_model_name)
            self.semantic_model.to(self.device)
            self.semantic_model.eval()
            logger.info(f"Semantic model loaded from {semantic_model_name}")
        except Exception as e:
            logger.warning(f"Semantic model failed: {e}")
        try:
            from ultralytics import YOLO
            self.instance_model = YOLO(instance_model_path)
            logger.info(f"Instance model loaded from {instance_model_path}")
        except Exception as e:
            logger.warning(f"Instance model failed: {e}")

    def semantic_segmentation(self, image: np.ndarray) -> np.ndarray:
        if self.semantic_model is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.semantic_model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode='bilinear', align_corners=False)
        sem_mask = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
        return sem_mask

    def instance_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        if self.instance_model is None:
            return []
        results = self.instance_model(image)
        masks = []
        for r in results:
            if r.masks is not None:
                for mask in r.masks.data.cpu().numpy():
                    masks.append(mask > 0.5)
        return masks

    def panoptic_segmentation(self, image: np.ndarray) -> np.ndarray:
        sem_mask = self.semantic_segmentation(image)
        instance_masks = self.instance_segmentation(image)
        panoptic_mask = sem_mask.copy()
        offset = 1000
        for i, mask in enumerate(instance_masks):
            panoptic_mask[mask] = offset + i
        return panoptic_mask

    def visualize(self, image: np.ndarray, panoptic_mask: np.ndarray) -> np.ndarray:
        vis = image.copy()
        unique_ids = np.unique(panoptic_mask)
        for uid in unique_ids:
            if uid < 1000:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            vis[panoptic_mask == uid] = color
        return cv2.addWeighted(image, 0.6, vis, 0.4, 0)