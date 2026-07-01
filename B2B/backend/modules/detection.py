"""
detection.py - تشخیص پلاک و خودرو با YOLO
"""

from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjectDetector:
    """کلاس تشخیص اشیاء (خودرو و پلاک) با YOLO"""
    
    def init(
        self,
        vehicle_model_path: str = 'models/yolov8n.pt',
        plate_model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cuda'
    ):
        """
        Args:
            vehicle_model_path: مسیر مدل تشخیص خودرو
            plate_model_path: مسیر مدل تشخیص پلاک
            conf_threshold: آستانه اطمینان
            iou_threshold: آستانه IOU
            device: دستگاه اجرا
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # بارگذاری مدل خودرو
        self.vehicle_model = YOLO(vehicle_model_path)
        self.vehicle_model.to(device)
        
        # بارگذاری مدل پلاک (اختیاری)
        self.plate_model = None
        if plate_model_path and Path(plate_model_path).exists():
            self.plate_model = YOLO(plate_model_path)
            self.plate_model.to(device)
            logger.info(f"مدل پلاک از {plate_model_path} بارگذاری شد")
        else:
            logger.warning("مدل پلاک بارگذاری نشد، از تشخیص درونی YOLO استفاده می‌شود")
        
        # لیست کلاس‌های خودرو در COCO
        self.vehicle_classes = [2, 3, 5, 6, 7]  # car, motorcycle, bus, truck, train
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        تشخیص خودروها در تصویر
        
        Returns:
            لیست تشخیص‌ها با کلیدهای: bbox, confidence, class_id, class_name
        """
        results = self.vehicle_model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                if cls in self.vehicle_classes or len(self.vehicle_classes) == 0:
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.vehicle_model.names.get(cls, 'vehicle')
                    })
        
        return detections
    
    def detect_plates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        تشخیص پلاک‌ها در تصویر
        
        Returns:
            لیست تشخیص‌های پلاک
        """
        if self.plate_model:
            results = self.plate_model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        else:
            results = self.vehicle_model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                # اگر کلاس پلاک باشد (فرض می‌کنیم کلاس 0 پلاک است)
                if self.plate_model or (not self.plate_model and cls in [0]):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': 'license_plate'
                    })
        
        return detections
    
    def detect_vehicles_and_plates(self, image: np.ndarray) -> Dict[str, Any]:
        """
        تشخیص همزمان خودروها و پلاک‌ها
        
        Returns:
            دیکشنری با کلیدهای vehicles, plates, matches
        """
        vehicles = self.detect_vehicles(image)
        plates = self.detect_plates(image)
        
        # تطابق پلاک‌ها با خودروها (بر اساس IoU)
        matches = []
        used_vehicles = set()
        
        for plate in plates:
            best_match = None
            best_iou = 0
            
            for i, vehicle in enumerate(vehicles):
                if i in used_vehicles:
                    continue
                
                iou = self._calculate_iou(plate['bbox'], vehicle['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                used_vehicles.add(best_match)
                matches.append({
                    'plate': plate,
                    'vehicle': vehicles[best_match],
                    'iou': best_iou
                })
        
        return {
            'vehicles': vehicles,
            'plates': plates,
            'matches': matches,
            'total_vehicles': len(vehicles),
            'total_plates': len(plates),
            'matched_count': len(matches)
        }
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """محاسبه IoU بین دو bounding box"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        رسم bounding box روی تصویر
        
        Args:
            image: تصویر اصلی
            detections: لیست تشخیص‌ها
            color: رنگ (BGR)
        
        Returns:
            تصویر با آنوتیشن
        """
        img = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            conf = det.get('confidence', 0)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det.get('class_name', 'plate')}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def extract_plate_crops(self, image: np.ndarray) -> List[np.ndarray]:
        """
        استخراج تصاویر برش خورده پلاک‌ها
        
        Returns:
            لیست تصاویر پلاک
        """
        plates = self.detect_plates(image)
        crops = []
        
        for plate in plates:
            x1, y1, x2, y2 = map(int, plate['bbox'])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        
        return crops
                        