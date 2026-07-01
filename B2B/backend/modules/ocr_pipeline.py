"""
ocr_pipeline.py - خط لوله OCR شامل پیش‌پردازش، تشخیص ناحیه متن و خواندن حروف
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class OCRPipeline:
    def init(self, use_easyocr: bool = True, use_tesseract: bool = False):
        self.use_easyocr = use_easyocr
        self.use_tesseract = use_tesseract
        self.reader = None
        if use_easyocr:
            try:
                import easyocr
                self.reader = easyocr.Reader(['fa', 'en'], gpu=True)
                logger.info("EasyOCR initialized")
            except ImportError:
                logger.warning("EasyOCR not installed")
                self.use_easyocr = False
        if use_tesseract:
            try:
                import pytesseract
                self.tesseract = pytesseract
                logger.info("Tesseract initialized")
            except ImportError:
                logger.warning("pytesseract not installed")
                self.use_tesseract = False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def extract_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        processed = self.preprocess(image)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 and h > 10 and w/h < 10:
                regions.append(image[y:y+h, x:x+w])
        return regions

    def read_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        if not self.use_easyocr or self.reader is None:
            return "", 0.0
        results = self.reader.readtext(image, paragraph=True, detail=1)
        if not results:
            return "", 0.0
        best = max(results, key=lambda x: x[2])
        return best[1], best[2]

    def read_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        if not self.use_tesseract:
            return "", 0.0
        text = self.tesseract.image_to_string(image, config='--oem 3 --psm 7 -l fas')
        return text.strip(), 0.7 if text else 0.0

    def read_plate(self, image: np.ndarray) -> Tuple[str, float]:
        processed = self.preprocess(image)
        if self.use_easyocr and self.reader:
            text, conf = self.read_easyocr(processed)
            if text:
                return text, conf
        if self.use_tesseract:
            text, conf = self.read_tesseract(processed)
            return text, conf
        return "", 0.0