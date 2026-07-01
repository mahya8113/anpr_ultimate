"""
advanced_ocr.py - OCR پیشرفته برای تشخیص حروف و اعداد فارسی پلاک
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PERSIAN_CHARS = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹'
ENGLISH_DIGITS = '0123456789'
ALL_CHARS = PERSIAN_CHARS + PERSIAN_DIGITS + ENGLISH_DIGITS
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(ALL_CHARS)}
IDX2CHAR = {i + 1: ch for i, ch in enumerate(ALL_CHARS)}
NUM_CLASSES = len(ALL_CHARS) + 1


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, input_height: int = 48, input_width: int = 160):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.cnn_output_height = input_height // 16
        self.cnn_output_channels = 512
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_channels * self.cnn_output_height,
            hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.cnn(x)
        features = features.permute(0, 3, 1, 2)
        features = features.reshape(batch_size, -1, features.size(1) * features.size(2))
        rnn_out, _ = self.rnn(features)
        return self.fc(rnn_out)


class AdvancedOCR:
    def __init__(self, use_easyocr: bool = True, use_crnn: bool = False, 
                 crnn_model_path: Optional[str] = None,
                 use_tesseract: bool = False, use_gpu: bool = True, 
                 languages: List[str] = None,
                 enable_preprocessing: bool = True):
        
        self.use_easyocr = use_easyocr
        self.use_crnn = use_crnn
        self.use_tesseract = use_tesseract
        self.enable_preprocessing = enable_preprocessing
        self.reader = None
        self.crnn_model = None
        self.tesseract_available = False
        self.persian_digits = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
        self.allowed_chars = set(PERSIAN_CHARS + PERSIAN_DIGITS + ENGLISH_DIGITS)
        
        if use_easyocr:
            try:
                import easyocr
                if languages is None: languages = ['fa', 'en']
                self.reader = easyocr.Reader(languages, gpu=use_gpu and torch.cuda.is_available())
                logger.info("EasyOCR loaded")
            except ImportError:
                logger.warning("EasyOCR not installed")
                self.use_easyocr = False
        if use_crnn and crnn_model_path and Path(crnn_model_path).exists():
            self._load_crnn_model(crnn_model_path)
        elif use_crnn:
            logger.warning("CRNN model not found, disabled")
            self.use_crnn = False
        if use_tesseract:
            try:
                import pytesseract
                self.tesseract = pytesseract
                self.tesseract_available = True
                logger.info("Tesseract loaded")
            except ImportError:
                logger.warning("pytesseract not installed")
                self.use_tesseract = False
    
    def _load_crnn_model(self, model_path: str):
        try:
            self.crnn_model = CRNN(NUM_CLASSES)
            state_dict = torch.load(model_path, map_location='cpu')
            self.crnn_model.load_state_dict(state_dict)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.crnn_model.to(device)
            self.crnn_model.eval()
            logger.info(f"CRNN model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load CRNN: {e}")
            self.use_crnn = False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if not self.enable_preprocessing:
            return image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape
        if w < 200:
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def read_with_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        if self.reader is None:
            return "", 0.0
        try:
            results = self.reader.readtext(image, paragraph=True, detail=1)
            if not results:
                return "", 0.0
            best = max(results, key=lambda x: x[2])
            text = best[1].translate(self.persian_digits)
            text = ''.join([c for c in text if c in self.allowed_chars])
            return text.strip(), best[2]
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0
    
    def read_plate(self, image: np.ndarray, method: str = 'auto') -> Tuple[str, float, Dict[str, Any]]:
        processed = self.preprocess(image)
        if self.use_easyocr and self.reader:
            text, conf = self.read_with_easyocr(processed)
            if text:
                return text, conf, {"methods_used": ["easyocr"], "best_method": "easyocr"}
        return "", 0.0, {"methods_used": [], "best_method": "none"}