"""
preprocessing.py - پیش‌پردازش تصاویر قبل از تشخیص
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """کلاس پیش‌پردازش تصاویر برای بهبود دقت تشخیص"""
    
    @staticmethod
    def enhance_contrast_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        افزایش کنتراست با CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            img: تصویر ورودی (BGR)
            clip_limit: محدودیت کنتراست
            tile_grid_size: اندازه سلول‌ها
        
        Returns:
            تصویر با کنتراست بهبود یافته
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        تصحیح گاما برای روشنایی تصویر
        
        Args:
            img: تصویر ورودی
            gamma: مقدار گاما (بیشتر از 1 = روشن‌تر، کمتر از 1 = تاریک‌تر)
        
        Returns:
            تصویر تصحیح شده
        """
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    
    @staticmethod
    def morphological_clean(img: np.ndarray, kernel_size: int = 3, operation: str = 'close') -> np.ndarray:
        """
        عملیات مورفولوژی برای حذف نویز و پر کردن حفره‌ها
        
        Args:
            img: تصویر ورودی
            kernel_size: اندازه کرنل
            operation: 'close', 'open', 'dilate', 'erode'
        
        Returns:
            تصویر پس از عملیات مورفولوژی
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'close':
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            return cv2.dilate(img, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(img, kernel, iterations=1)
        else:
            return img
    
    @staticmethod
    def resize(img: np.ndarray, target_size: Tuple[int, int], keep_aspect: bool = True) -> np.ndarray:
        """
        تغییر اندازه تصویر
        
        Args:
            img: تصویر ورودی
            target_size: اندازه هدف (width, height)
            keep_aspect: حفظ نسبت ابعاد (با padding)
        
        Returns:
            تصویر تغییر اندازه یافته
        """
        if keep_aspect:
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h))
            
            # اضافه کردن padding
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            
            padded = cv2.copyMakeBorder(resized, pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return padded
        else:
            return cv2.resize(img, target_size)
    
    @staticmethod
    def normalize(img: np.ndarray, to_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        نرمال‌سازی مقادیر پیکسل‌ها
        
        Args:
            img: تصویر ورودی
            to_range: بازه هدف (min, max)
            Returns:
            تصویر نرمال‌سازی شده
        """
        img_float = img.astype(np.float32)
        img_norm = cv2.normalize(img_float, None, to_range[0], to_range[1], cv2.NORM_MINMAX)
        return img_norm
    
    @staticmethod
    def remove_shadow(img: np.ndarray) -> np.ndarray:
        """
        حذف سایه از تصویر
        
        Args:
            img: تصویر ورودی
        
        Returns:
            تصویر بدون سایه
        """
        rgb_planes = cv2.split(img)
        result_planes = []
        
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        
        return cv2.merge(result_planes)
    
    @staticmethod
    def auto_rotate(img: np.ndarray) -> np.ndarray:
        """
        چرخش خودکار تصویر بر اساس EXIF (برای تصاویر گوشی)
        
        Args:
            img: تصویر ورودی
        
        Returns:
            تصویر چرخش یافته
        """
        # در صورت نیاز می‌توان با PIL EXIF را خواند
        # این یک پیاده‌سازی ساده است
        return img
    
    @staticmethod
    def full_preprocess(img: np.ndarray, for_detection: bool = True) -> np.ndarray:
        """
        خط لوله کامل پیش‌پردازش
        
        Args:
            img: تصویر ورودی
            for_detection: آیا برای تشخیص است؟ (اگر بله، کنتراست بالا می‌رود)
        
        Returns:
            تصویر پیش‌پردازش شده
        """
        # مرحله 1: حذف سایه
        img = Preprocessor.remove_shadow(img)
        
        # مرحله 2: افزایش کنتراست
        if for_detection:
            img = Preprocessor.enhance_contrast_clahe(img, clip_limit=2.5)
        else:
            img = Preprocessor.enhance_contrast_clahe(img, clip_limit=1.5)
        
        # مرحله 3: تصحیح گاما
        img = Preprocessor.gamma_correction(img, gamma=1.2)
        
        # مرحله 4: عملیات مورفولوژی سبک
        img = Preprocessor.morphological_clean(img, kernel_size=3, operation='close')
        
        return img
    