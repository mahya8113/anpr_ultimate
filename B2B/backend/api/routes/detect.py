"""
detect.py - مسیرهای تشخیص پلاک (تصویر، ویدئو، استریم)
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Optional
import cv2
import numpy as np
import base64
from datetime import datetime

from modules.detection import ObjectDetector
from modules.advanced_ocr import AdvancedOCR
from modules.image_formation import ImageFormation
from modules.preprocessing import Preprocessor
from modules.geometry import GeometryCorrector
from core.cache import DetectionCache
from core.security import get_current_user

router = APIRouter(prefix="/detect", tags=["تشخیص"])

# بارگذاری مدل‌ها (در سطح ماژول یا می‌توان در startup انجام داد)
detector = ObjectDetector()
ocr = AdvancedOCR()


@router.post("/image")
async def detect_from_image(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """تشخیص پلاک از تصویر آپلودی"""
    # بررسی فرمت فایل
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "فایل باید از نوع تصویر باشد")
    
    # خواندن تصویر
    img = await ImageFormation.from_upload(file)
    
    # کش: بررسی وجود نتیجه مشابه
    img_bytes = await file.read()
    cached = await DetectionCache.get(img_bytes)
    if cached:
        return cached
    
    # پیش‌پردازش
    processed = Preprocessor.full_preprocess(img)
    
    # تشخیص پلاک
    plates = detector.detect_plates(processed)
    results = []
    for plate in plates:
        # برش و تصحیح پرسپکتیو
        bbox = plate['bbox']
        crop = GeometryCorrector.crop_plate_region(processed, bbox)
        warped = GeometryCorrector.straighten_plate(crop)
        # OCR
        text, conf, info = ocr.read_plate(warped)
        results.append({
            "bbox": bbox,
            "plate_text": text,
            "confidence": plate['confidence'],
            "ocr_confidence": conf
        })
    
    response = {
        "success": True,
        "plates": results,
        "num_plates": len(results),
        "timestamp": datetime.now().isoformat()
    }
    
    # ذخیره در کش
    await DetectionCache.set(img_bytes, response)
    
    return response


@router.post("/image-base64")
async def detect_from_base64(
    image_base64: str = Form(...),
    current_user=Depends(get_current_user)
):
    """تشخیص پلاک از تصویر base64"""
    img = ImageFormation.from_base64(image_base64)
    processed = Preprocessor.full_preprocess(img)
    plates = detector.detect_plates(processed)
    results = []
    for plate in plates:
        bbox = plate['bbox']
        crop = GeometryCorrector.crop_plate_region(processed, bbox)
        text, conf, _ = ocr.read_plate(crop)
        results.append({
            "bbox": bbox,
            "plate_text": text,
            "confidence": plate['confidence']
        })
    return {"plates": results, "num_plates": len(results)}


@router.post("/url")
async def detect_from_url(
    url: str = Form(...),
    current_user=Depends(get_current_user)
):
    """تشخیص پلاک از آدرس URL تصویر"""
    img = await ImageFormation.from_url(url)
    processed = Preprocessor.full_preprocess(img)
    plates = detector.detect_plates(processed)
    results = []
    for plate in plates:
        bbox = plate['bbox']
        crop = GeometryCorrector.crop_plate_region(processed, bbox)
        text, conf, _ = ocr.read_plate(crop)
        results.append({"plate_text": text, "confidence": plate['confidence']})
    return {"plates": results}