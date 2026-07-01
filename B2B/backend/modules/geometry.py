"""
geometry.py - تبدیلات هندسی و تصحیح پرسپکتیو پلاک
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class GeometryCorrector:
    """کلاس تصحیح هندسی و پرسپکتیو پلاک"""
    
    @staticmethod
    def four_point_transform(image: np.ndarray, pts: np.ndarray, output_size: Tuple[int, int] = (320, 100)) -> np.ndarray:
        """
        تبدیل پرسپکتیو با استفاده از 4 نقطه
        
        Args:
            image: تصویر ورودی
            pts: چهار نقطه (چهار گوشه پلاک)
            output_size: اندازه خروجی (عرض, ارتفاع)
        
        Returns:
            تصویر تصحیح شده پلاک
        """
        # مرتب‌سازی نقاط
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # بالا-چپ
        rect[2] = pts[np.argmax(s)]  # پایین-راست
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # بالا-راست
        rect[3] = pts[np.argmax(diff)]  # پایین-چپ
        
        (tl, tr, br, bl) = rect
        
        # محاسبه عرض و ارتفاع هدف
        widthA = np.sqrt(((br[0] - bl[0])**  2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0])**  2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # ابعاد خروجی
        if output_size:
            dst_w, dst_h = output_size
        else:
            dst_w, dst_h = maxWidth, maxHeight
        
        dst = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
        
        return warped
    
    @staticmethod
    def auto_correct_perspective(image: np.ndarray) -> np.ndarray:
        """
        تصحیح خودکار پرسپکتیو با تشخیص چهارگوش بزرگ در تصویر
        
        Args:
            image: تصویر ورودی
        
        Returns:
            تصویر تصحیح شده
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # پیدا کردن بزرگترین کانتور
        if not contours:
            return image
        
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        
        if len(approx) == 4:
            return GeometryCorrector.four_point_transform(image, approx.reshape(4, 2))
        
        return image
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        چرخش تصویر حول مرکز
        
        Args:
            image: تصویر ورودی
            angle: زاویه چرخش (درجه)
            center: مرکز چرخش (اگر None باشد، مرکز تصویر)
        
        Returns:
            تصویر چرخش یافته
        """
        h, w = image.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def resize_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        تغییر اندازه با حفظ نسبت ابعاد و اضافه کردن padding سیاه
        Args:
            image: تصویر ورودی
            target_size: اندازه هدف (عرض, ارتفاع)
        
        Returns:
            تصویر تغییر اندازه یافته با padding
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # ایجاد تصویر خالی با padding
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # محاسبه موقعیت قرارگیری
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    
    @staticmethod
    def crop_plate_region(image: np.ndarray, bbox: List[int], margin: int = 10) -> np.ndarray:
        """
        برش ناحیه پلاک از تصویر با حاشیه اضافی
        
        Args:
            image: تصویر اصلی
            bbox: مختصات [x1, y1, x2, y2]
            margin: حاشیه اضافی (پیکسل)
        
        Returns:
            تصویر برش خورده پلاک
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # اعمال حاشیه
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def straighten_plate(plate_image: np.ndarray) -> np.ndarray:
        """
        صاف‌سازی پلاک با استفاده از تشخیص لبه‌ها و چرخش
        
        Args:
            plate_image: تصویر پلاک
        
        Returns:
            تصویر پلاک صاف شده
        """
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return plate_image
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            rotated = GeometryCorrector.rotate_image(plate_image, -median_angle)
            return rotated
        
        return plate_image