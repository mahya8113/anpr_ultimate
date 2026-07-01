"""
ocr_service.py - سرویس تشخیص حروف و اعداد فارسی پلاک
پشتیبانی از: EasyOCR، CRNN اختصاصی، پیش‌پردازش پیشرفته
"""

import cv2
import numpy as np
import torch
import re
from typing import Optional, Tuple, Dict, Any
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class PersianOCRService:
    """
    سرویس OCR برای تشخیص حروف و اعداد فارسی پلاک خودرو
    پشتیبانی از EasyOCR (پیش‌فرض) و مدل اختصاصی CRNN
    """
    
    def init(
        self,
        use_easyocr: bool = True,
        crnn_model_path: Optional[str] = None,
        languages: list = None,
        use_gpu: bool = True,
        enable_preprocessing: bool = True
    ):
        """
        Args:
            use_easyocr: استفاده از EasyOCR (پیش‌فرض)
            crnn_model_path: مسیر مدل اختصاصی CRNN
            languages: لیست زبان‌ها
            use_gpu: استفاده از GPU
            enable_preprocessing: فعال‌سازی پیش‌پردازش
        """
        self.use_easyocr = use_easyocr
        self.enable_preprocessing = enable_preprocessing
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if languages is None:
            languages = ['fa', 'en', 'ar']
        
        # نگاشت اعداد فارسی به انگلیسی
        self.persian_digits = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
        
        # حروف و اعداد مجاز پلاک ایران
        self.allowed_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی' + '۰۱۲۳۴۵۶۷۸۹' + '0123456789')
        
        # بارگذاری EasyOCR
        if use_easyocr:
            try:
                import easyocr
                self.reader = easyocr.Reader(languages, gpu=self.use_gpu)
                logger.info("EasyOCR با موفقیت بارگذاری شد")
            except Exception as e:
                logger.error(f"خطا در بارگذاری EasyOCR: {e}")
                self.reader = None
        else:
            self.reader = None
        
        # بارگذاری مدل اختصاصی CRNN
        self.crnn_model = None
        if crnn_model_path and Path(crnn_model_path).exists():
            self._load_crnn_model(crnn_model_path)
    
    def _load_crnn_model(self, model_path: str):
        """بارگذاری مدل اختصاصی CRNN"""
        try:
            import torch.nn as nn
            
            # تعریف معماری ساده CRNN
            class CRNN(nn.Module):
                def init(self, num_classes=35):
                    super().__init__()
                    self.cnn = nn.Sequential(
                        nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    )
                    self.lstm = nn.LSTM(256 * 6, 128, bidirectional=True, batch_first=True)
                    self.fc = nn.Linear(256, num_classes)
                
                def forward(self, x):
                    features = self.cnn(x)
                    features = features.permute(0, 3, 1, 2)
                    features = features.reshape(features.size(0), features.size(1), -1)
                    lstm_out, _ = self.lstm(features)
                    return self.fc(lstm_out)
            
            self.crnn_model = CRNN()
            self.crnn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            if self.use_gpu:
                self.crnn_model.cuda()
            self.crnn_model.eval()
            logger.info(f"مدل CRNN از {model_path} بارگذاری شد")
            
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل CRNN: {e}")
            self.crnn_model = None
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        پیش‌پردازش تصویر پلاک برای بهبود OCR
        
        Args:
        image: تصویر پلاک
        
        Returns:
            تصویر پیش‌پردازش شده
        """
        if not self.enable_preprocessing:
            return image
        
        # تبدیل به خاکستری
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # افزایش اندازه برای دقت بهتر
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        # کاهش نویز
        gray = cv2.medianBlur(gray, 3)
        
        # افزایش کنتراست
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # باینری کردن (اختیاری)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def read_with_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        خواندن پلاک با EasyOCR
        
        Args:
            image: تصویر پلاک
        
        Returns:
            (متن تشخیص داده شده, اطمینان)
        """
        if self.reader is None:
            return "", 0.0
        
        try:
            results = self.reader.readtext(image, paragraph=True, detail=1)
            
            if not results:
                return "", 0.0
            
            # بهترین نتیجه با بیشترین اطمینان
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1]
            confidence = best_result[2]
            
            # تبدیل اعداد فارسی به انگلیسی
            text = text.translate(self.persian_digits)
            
            # فیلتر کاراکترهای مجاز
            text = ''.join([c for c in text if c in self.allowed_chars or c == ' '])
            text = text.strip()
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"خطا در EasyOCR: {e}")
            return "", 0.0
    
    def read_with_crnn(self, image: np.ndarray) -> Tuple[str, float]:
        """
        خواندن پلاک با مدل اختصاصی CRNN
        
        Args:
            image: تصویر پلاک
        
        Returns:
            (متن تشخیص داده شده, اطمینان)
        """
        if self.crnn_model is None:
            return "", 0.0
        
        try:
            # پیش‌پردازش برای CRNN
            processed = self.preprocess(image)
            processed = cv2.resize(processed, (160, 48))
            processed = processed.astype(np.float32) / 255.0
            tensor = torch.tensor(processed).unsqueeze(0).unsqueeze(0)
            
            if self.use_gpu:
                tensor = tensor.cuda()
            
            with torch.no_grad():
                output = self.crnn_model(tensor)
                preds = torch.softmax(output, dim=2)
                preds = torch.argmax(preds, dim=2).squeeze().cpu().numpy()
            
            # تبدیل ایندکس‌ها به متن
            chars = '0123456789ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
            text = ''.join([chars[p-1] for p in preds if p > 0])
            
            # حذف کاراکترهای تکراری متوالی
            result = ''
            prev = ''
            for ch in text:
                if ch != prev:
                    result += ch
                    prev = ch
            
            confidence = 0.85  # مقدار تخمینی
            
            return result, confidence
            
        except Exception as e:
            logger.error(f"خطا در CRNN: {e}")
            return "", 0.0
    
    def read_plate(self, image: np.ndarray) -> Tuple[str, float]:
        """
        خواندن پلاک با بهترین روش موجود
        
        Args:
            image: تصویر پلاک
        
        Returns:
            (متن تشخیص داده شده, اطمینان)
        """
        # پیش‌پردازش
        processed = self.preprocess(image)
        # OCR با روش انتخابی
        if self.use_easyocr and self.reader:
            text, confidence = self.read_with_easyocr(processed)
        elif self.crnn_model:
            text, confidence = self.read_with_crnn(processed)
        else:
            logger.warning("هیچ OCR ای در دسترس نیست")
            return "", 0.0
        
        # اعتبارسنجی و پست‌پردازش
        text = self.postprocess(text)
        
        return text, confidence
    
    def postprocess(self, text: str) -> str:
        """
        پست‌پردازش متن تشخیص داده شده
        
        Args:
            text: متن خام OCR
        
        Returns:
            متن تصحیح شده
        """
        if not text:
            return ""
        
        # حذف فاصله‌های اضافی
        text = ' '.join(text.split())
        
        # حذف کاراکترهای غیرمجاز
        text = ''.join([c for c in text if c in self.allowed_chars])
        
        # فرمت استاندارد پلاک ایران (اختیاری)
        if len(text) == 7 and text.isdigit():
            # 7 رقم → فرمت 1234567
            pass
        elif len(text) >= 5:
            # حذف حروف تکراری
            result = ''
            prev = ''
            for ch in text:
                if ch != prev:
                    result += ch
                    prev = ch
            text = result
        
        return text.upper()
    
    def read_plate_batch(self, images: list) -> list:
        """
        خواندن دسته‌ای پلاک‌ها
        
        Args:
            images: لیست تصاویر پلاک
        
        Returns:
            لیست نتایج (متن, اطمینان)
        """
        results = []
        for image in images:
            text, confidence = self.read_plate(image)
            results.append((text, confidence))
        return results
    
    async def read_plate_async(self, image: np.ndarray) -> Tuple[str, float]:
        """
        خواندن ناهمگام پلاک
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_plate, image)
    