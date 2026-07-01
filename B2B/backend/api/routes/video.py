"""
video.py - مسیرهای پردازش ویدئو و استریم
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Dict, Any
import cv2
import tempfile
import os
from datetime import datetime

from modules.detection import ObjectDetector
from modules.advanced_ocr import AdvancedOCR
from modules.multi_tracking import MultiTracker
from core.security import get_current_user

router = APIRouter(prefix="/video", tags=["ویدئو"])

detector = ObjectDetector()
ocr = AdvancedOCR()
tracker = MultiTracker()


@router.post("/process")
async def process_video(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """پردازش ویدئو آپلودی و استخراج پلاک‌ها در فریم‌ها"""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "فایل باید از نوع ویدئو باشد")
    
    # ذخیره موقت فایل
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:  # پردازش هر 5 فریم
            plates = detector.detect_plates(frame)
            for plate in plates:
                x1, y1, x2, y2 = map(int, plate['bbox'])
                crop = frame[y1:y2, x1:x2]
                text, conf, _ = ocr.read_plate(crop)
                detections.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "plate_text": text,
                    "confidence": plate['confidence'],
                    "ocr_confidence": conf,
                    "bbox": plate['bbox']
                })
        frame_count += 1
    
    cap.release()
    os.unlink(tmp_path)
    
    return {
        "total_frames": frame_count,
        "fps": fps,
        "detections": detections,
        "num_detections": len(detections),
        "processing_time": datetime.now().isoformat()
    }


@router.post("/track")
async def track_in_video(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """ردیابی خودروها در ویدئو به همراه تشخیص پلاک"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracks_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # تشخیص خودروها و پلاک‌ها
        vehicles = detector.detect_vehicles(frame)
        plates = detector.detect_plates(frame)
        
        # ترکیب تشخیص‌ها برای ردیاب
        detections = vehicles + plates
        tracks = tracker.update(detections, frame)
        
        for track in tracks:
            tracks_data.append({
                "frame": frame_count,
                "track_id": track.track_id,
                "bbox": track.to_ltrb(),
                "timestamp": frame_count / fps
            })
        
        frame_count += 1
    
    cap.release()
    os.unlink(tmp_path)
    
    return {
        "total_frames": frame_count,
        "fps": fps,
        "tracks": tracks_data,
        "unique_tracks": len(set(t['track_id'] for t in tracks_data))
    }