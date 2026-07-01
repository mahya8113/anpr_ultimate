"""
detection_service.py - سرویس تشخیص پلاک با استفاده از YOLOv8
شامل: تشخیص خودرو، تشخیص پلاک، استخراج ویژگی‌ها
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """نتیجه تشخیص پلاک"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    features: Dict[str, Any] = field(default_factory=dict)


class DetectionService:
    """
    سرویس تشخیص پلاک با استفاده از YOLOv8
    پشتیبانی از: تشخیص خودرو، تشخیص پلاک، پیش‌پردازش تصویر
    """
    
    def init(
        self,
        model_path: str = "models/yolov8n.pt",
        plate_model_path: str = "models/yolov8n_plate_v1.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        enable_preprocessing: bool = True
    ):
        """
        Args:
            model_path: مسیر مدل اصلی YOLO
            plate_model_path: مسیر مدل تشخیص پلاک
            conf_threshold: آستانه اطمینان
            iou_threshold: آستانه IOU برای NMS
            device: دستگاه اجرا ('cuda' یا 'cpu')
            enable_preprocessing: فعال‌سازی پیش‌پردازش
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_preprocessing = enable_preprocessing
        
        # بارگذاری مدل‌ها
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"مدل اصلی از {model_path} بارگذاری شد روی {self.device}")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل اصلی: {e}")
            self.model = None
        
        try:
            self.plate_model = YOLO(plate_model_path)
            self.plate_model.to(self.device)
            logger.info(f"مدل پلاک از {plate_model_path} بارگذاری شد روی {self.device}")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل پلاک: {e}")
            self.plate_model = None
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        پیش‌پردازش تصویر برای بهبود دقت تشخیص
        
        Args:
            image: تصویر ورودی (BGR)
        
        Returns:
            تصویر پیش‌پردازش شده
        """
        if not self.enable_preprocessing:
            return image
        
        # بهبود کنتراست با CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # کاهش نویز
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # عملیات مورفولوژی
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return morphed
    
    def detect_vehicles(self, image: np.ndarray) -> List[DetectionResult]:
        """
        تشخیص خودروها در تصویر
        
        Args:
            image: تصویر ورودی
        
        Returns:
            لیست نتایج تشخیص خودرو
        """
        if self.model is None:
            logger.warning("مدل اصلی بارگذاری نشده است")
            return []
        
        processed = self.preprocess(image)
        results = self.model(processed, conf=self.conf_threshold, iou=self.iou_threshold)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                class_name = self.model.names.get(cls, "unknown")
                
                detections.append(DetectionResult(
                    bbox=box.astype(int).tolist(),
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name=class_name
                ))
        
        return detections
    
    def detect_plates(self, image: np.ndarray) -> List[DetectionResult]:
        """
        تشخیص پلاک‌ها در تصویر
        
        Args:
            image: تصویر ورودی
        
        Returns:
            لیست نتایج تشخیص پلاک
        """
        if self.plate_model is None:
            logger.warning("مدل پلاک بارگذاری نشده است")
            return self._fallback_plate_detection(image)
        
        processed = self.preprocess(image)
        results = self.plate_model(processed, conf=self.conf_threshold, iou=self.iou_threshold)
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                detections.append(DetectionResult(
                    bbox=box.astype(int).tolist(),
                    confidence=float(conf),
                    class_id=0,
                    class_name="license_plate"
                ))
        
        return detections
    
    def _fallback_plate_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """
        روش جایگزین برای تشخیص پلاک (در صورت عدم وجود مدل)
        استفاده از پردازش تصویر سنتی
        """
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # جستجوی مستطیل‌ها با نسبت ابعاد مشابه پلاک
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # نسبت ابعاد پلاک معمولاً بین 2 تا 5 است
                if 2 < aspect_ratio < 5 and w > 50 and h > 20:
                    detections.append(DetectionResult(
                        bbox=[x, y, x + w, y + h],
                        confidence=0.6,
                        class_id=0,
                        class_name="license_plate"
                    ))
        
        return detections
    
    def extract_plate_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        استخراج نواحی پلاک از تصویر
        
        Args:
            image: تصویر اصلی
        
        Returns:
            لیست تصاویر برش خورده پلاک
        """
        plates = self.detect_plates(image)
        plate_images = []
        
        for plate in plates:
            x1, y1, x2, y2 = plate.bbox
            plate_crop = image[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                # تغییر اندازه برای OCR
                plate_resized = cv2.resize(plate_crop, (320, 100))
                plate_images.append(plate_resized)
        
        return plate_images
    
    def detect_vehicles_and_plates(self, image: np.ndarray) -> Dict[str, Any]:
        """
        تشخیص همزمان خودروها و پلاک‌ها
        
        Args:
            image: تصویر ورودی
        
        Returns:
            دیکشنری شامل خودروها و پلاک‌های تشخیص داده شده
        """
        vehicles = self.detect_vehicles(image)
        plates = self.detect_plates(image)
        
        # تطابق پلاک با خودرو (بر اساس موقعیت)
        matched_results = []
        for plate in plates:
            best_match = None
            best_iou = 0
            
            for vehicle in vehicles:
                iou = self._calculate_iou(plate.bbox, vehicle.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = vehicle
            
            matched_results.append({
                "plate": plate,
                "vehicle": best_match,
                "iou": best_iou
            })
        
        return {
            "vehicles": [self._result_to_dict(v) for v in vehicles],
            "plates": [self._result_to_dict(p) for p in plates],
            "matches": [
                {
                    "plate": self._result_to_dict(m["plate"]),
                    "vehicle": self._result_to_dict(m["vehicle"]) if m["vehicle"] else None,
                    "iou": m["iou"]
                }
                for m in matched_results
            ],
            "total_vehicles": len(vehicles),
            "total_plates": len(plates)
        }
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
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
        
        return intersection / union if union > 0 else 0.0
    
    def _result_to_dict(self, result: DetectionResult) -> Dict[str, Any]:
        """تبدیل نتیجه تشخیص به دیکشنری"""
        return {
            "bbox": result.bbox,
            "confidence": result.confidence,
            "class_id": result.class_id,
            "class_name": result.class_name,
            "plate_text": result.plate_text,
            "timestamp": result.timestamp.isoformat()
        }
    
    async def process_image_async(self, image: np.ndarray) -> Dict[str, Any]:
        """
        پردازش ناهمگام تصویر
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.detect_vehicles_and_plates, image)
        return result
    
    def draw_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        رسم bounding box روی تصویر
        """
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0) if det.class_name == "license_plate" else (255, 0, 0)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det.class_name}: {det.confidence:.2f}"
            if det.plate_text:
                label = f"{det.plate_text} ({det.confidence:.2f})"
            
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img
    
        