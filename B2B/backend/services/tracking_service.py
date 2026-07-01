"""
tracking_service.py - سرویس ردیابی چند شیء با DeepSORT
قابلیت‌ها: ردیابی خودروها، ذخیره مسیر، تشخیص ورود/خروج
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("DeepSORT در دسترس نیست. نصب: pip install deep-sort-realtime")


@dataclass
class Track:
    """اطلاعات یک شیء ردیابی شده"""
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    trajectory: List[List[int]] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    plate_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
            "trajectory": self.trajectory[-10:],  # آخرین 10 موقعیت
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "plate_text": self.plate_text,
            "metadata": self.metadata
        }


class TrackingService:
    """
    سرویس ردیابی چند شیء با استفاده از DeepSORT
    """
    
    def init(
        self,
        max_age: int = 30,
        n_init: int = 3,
        nn_budget: int = 100,
        use_custom_model: bool = False
    ):
        """
        Args:
            max_age: حداکثر عمر یک track بدون به‌روزرسانی
            n_init: تعداد فریم‌های اولیه برای تأیید track
            nn_budget: بودجه حافظه برای ویژگی‌ها
            use_custom_model: استفاده از مدل سفارشی برای استخراج ویژگی
        """
        self.max_age = max_age
        self.n_init = n_init
        
        # مقداردهی DeepSORT
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nn_budget=nn_budget,
                embedder="mobilenet" if not use_custom_model else "custom"
            )
        else:
            self.tracker = None
            logger.warning("سرویس ردیابی بدون DeepSORT کار می‌کند (عملکرد محدود)")
        
        self.tracks: Dict[int, Track] = {}
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # آستانه‌های ورود/خروج
        self.entry_zones = []
        self.exit_zones = []
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Track]:
        """
        به‌روزرسانی ردیاب با تشخیص‌های جدید
        
        Args:
            detections: لیست تشخیص‌ها (هر کدام شامل bbox, confidence, class_id)
            frame: فریم تصویر
        
        Returns:
            لیست trackهای به‌روز شده
        """
        if self.tracker is None:
            return self._manual_update(detections)
        
        # تبدیل تشخیص‌ها به فرمت DeepSORT
        deepsort_detections = []
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            confidence = det.get('confidence', 0.5)
            class_id = det.get('class_id', 0)
            
            # تبدیل به فرمت [left, top, width, height]
            ltrb = bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            deepsort_detections.append(([bbox[0], bbox[1], width, height], confidence, class_id))
            # به‌روزرسانی ردیاب
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        # تبدیل به فرمت داخلی
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            
            # به‌روزرسانی یا ایجاد track جدید
            if track_id in self.tracks:
                old_track = self.tracks[track_id]
                old_track.bbox = bbox
                old_track.age = track.age
                old_track.hits = track.hits
                old_track.time_since_update = track.time_since_update
                old_track.trajectory.append([bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2])
                old_track.last_seen = datetime.now()
                results.append(old_track)
            else:
                new_track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    class_id=0,
                    confidence=track.get_det_conf() or 0.5,
                    age=track.age,
                    hits=track.hits,
                    trajectory=[[bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]]
                )
                self.tracks[track_id] = new_track
                results.append(new_track)
            
            self.track_history[track_id].append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
        
        # حذف trackهای قدیمی
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return results
    
    def _manual_update(self, detections: List[Dict]) -> List[Track]:
        """
        به‌روزرسانی دستی (بدون DeepSORT)
        """
        # ساده: هر تشخیص جدید یک track جدید است
        results = []
        for det in detections:
            track_id = len(self.tracks) + 1
            bbox = det['bbox']
            
            track = Track(
                track_id=track_id,
                bbox=bbox,
                class_id=det.get('class_id', 0),
                confidence=det.get('confidence', 0.5)
            )
            self.tracks[track_id] = track
            results.append(track)
        
        return results
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """دریافت track با شناسه مشخص"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[Track]:
        """دریافت همه trackهای فعال"""
        return list(self.tracks.values())
    
    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """دریافت مسیر حرکت یک شیء"""
        return self.track_history.get(track_id, [])
    
    def set_entry_zones(self, zones: List[List[int]]):
        """تنظیم مناطق ورود"""
        self.entry_zones = zones
    
    def set_exit_zones(self, zones: List[List[int]]):
        """تنظیم مناطق خروج"""
        self.exit_zones = zones
    
    def check_entry_exit(self, track: Track) -> Tuple[bool, bool]:
        """
        بررسی ورود یا خروج از مناطق تعیین شده
        
        Returns:
            (is_entry, is_exit)
        """
        if not track.trajectory:
            return False, False
        
        center = track.trajectory[-1]
        is_entry = any(self._point_in_zone(center, zone) for zone in self.entry_zones)
        is_exit = any(self._point_in_zone(center, zone) for zone in self.exit_zones)
        
        return is_entry, is_exit
    
    def _point_in_zone(self, point: Tuple[int, int], zone: List[int]) -> bool:
        """بررسی نقطه درون منطقه مستطیلی"""
        x, y = point
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def assign_plate_to_track(self, track_id: int, plate_text: str):
        """اختصاص پلاک به یک track"""
        if track_id in self.tracks:
            self.tracks[track_id].plate_text = plate_text
            self.tracks[track_id].metadata["plate_assigned_at"] = datetime.now().isoformat()
    
    def get_tracks_with_plate(self) -> List[Track]:
        """دریافت trackهایی که پلاک به آنها اختصاص داده شده است"""
        return [t for t in self.tracks.values() if t.plate_text]
    
    def clear_old_tracks(self, max_age_seconds: int = 60):
        """حذف trackهای قدیمی"""
        now = datetime.now()
        to_remove = []
        
        for track_id, track in self.tracks.items():
            age_seconds = (now - track.last_seen).total_seconds()
            if age_seconds > max_age_seconds:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """آمار ردیابی"""
        return {
            "active_tracks": len(self.tracks),
            "total_tracks_ever": len(self.track_history),
            "tracks_with_plate": len(self.get_tracks_with_plate()),
            "max_age": self.max_age,
            "entry_zones_count": len(self.entry_zones),
            "exit_zones_count": len(self.exit_zones)
        }
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """
        رسم مسیر حرکت روی فریم
        
        Args:
            frame: فریم تصویر
        
        Returns:
            تصویر با مسیرهای رسم شده
        """
        img = frame.copy()
        
        for track in self.tracks.values():
            if len(track.trajectory) < 2:
                continue
            
            # رسم مسیر
            np.array(track.trajectory[-30:], dtype=np.int32)
            cv2.polylines(img, [points], False, (0, 255, 0), 2)  # noqa: F821
            
            # رسم bounding box
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # نمایش track_id
            label = f"ID: {track.track_id}"
            if track.plate_text:
                label += f" | {track.plate_text}"
            
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # رسم مناطق ورود/خروج
        for zone in self.entry_zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "ENTRY", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for zone in self.exit_zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "EXIT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img
