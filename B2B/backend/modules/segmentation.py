"""
segmentation.py - قطعه‌بندی اشیاء (Instance Segmentation) با YOLO-seg و CRF
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Segmenter:
    def init(self, model_path: str = 'yolov8n-seg.pt', device: str = 'cuda'):
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info(f"Segmentation model loaded from {model_path}")

    def instance_segmentation(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        results = self.model(image, conf=conf_threshold, verbose=False)
        segments = []
        for r in results:
            if r.masks is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data.cpu().numpy()
            for box, conf, cls, mask in zip(boxes, confs, classes, masks):
                segments.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.model.names.get(cls, 'object'),
                    'mask': mask
                })
        return segments

    def apply_crf(self, image: np.ndarray, soft_mask: np.ndarray, n_iter: int = 5) -> np.ndarray:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        h, w = soft_mask.shape[:2]
        n_classes = 2
        unary = unary_from_softmax(np.stack([1 - soft_mask, soft_mask], axis=0))
        d = dcrf.DenseCRF2D(w, h, n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
        Q = d.inference(n_iter)
        return np.argmax(Q, axis=0).reshape((h, w))

    def get_largest_mask(self, masks: List[np.ndarray]) -> Optional[np.ndarray]:
        if not masks:
            return None
        sizes = [np.sum(m) for m in masks]
        return masks[np.argmax(sizes)]

    def overlay_masks(self, image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
        overlay = image.copy()
        for mask in masks:
            color = np.random.randint(0, 255, 3).tolist()
            colored = np.zeros_like(image)
            colored[mask > 0.5] = color
            overlay = cv2.addWeighted(overlay, alpha, colored, 1 - alpha, 0)
        return overlay

    def segment_characters(self, plate_image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 10:
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                masks.append(mask)
        return masks