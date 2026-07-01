"""
restoration.py - بازسازی و بهبود کیفیت تصاویر مخدوش
"""

import cv2
import numpy as np
from scipy.signal import convolve2d
from typing import Optional, Tuple  
import logging

logger = logging.getLogger(__name__)


class ImageRestorer:
    """کلاس بازسازی تصاویر (حذف نویز، رفع تارشدگی، وینر فیلتر)"""
    
    @staticmethod
    def wiener_filter(img: np.ndarray, kernel_size: int = 5, K: float = 0.01) -> np.ndarray:
        """
        فیلتر وینر برای رفع تارشدگی و کاهش نویز
        
        Args:
            img: تصویر ورودی
            kernel_size: اندازه کرنل
            K: پارامتر وینر
        
        Returns:
            تصویر بازسازی شده
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        img_fft = np.fft.fft2(img_gray)
        kernel_fft = np.fft.fft2(kernel, s=img_gray.shape)
        kernel_fft_conj = np.conj(kernel_fft)
        
        result_fft = (kernel_fft_conj / (kernel_fft * kernel_fft_conj + K)) * img_fft
        restored = np.fft.ifft2(result_fft).real.astype(np.uint8)
        
        if len(img.shape) == 3:
            restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        
        return restored
    
    @staticmethod
    def non_local_means_denoise(img: np.ndarray, h: int = 10, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """
        حذف نویز با روش Non-Local Means
        
        Args:
            img: تصویر ورودی
            h: قدرت فیلتر (بیشتر = نویز کمتر)
            template_window_size: اندازه پنجره الگو
            search_window_size: اندازه پنجره جستجو
        
        Returns:
            تصویر بدون نویز
        """
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window_size, search_window_size)
        else:
            return cv2.fastNlMeansDenoising(img, None, h, template_window_size, search_window_size)
    
    @staticmethod
    def deblur_motion(img: np.ndarray, kernel_size: int = 15, angle: float = 0) -> np.ndarray:
        """
        رفع تارشدگی ناشی از حرکت دوربین
        
        Args:
            img: تصویر ورودی
            kernel_size: اندازه کرنل حرکت
            angle: زاویه حرکت (درجه)
        
        Returns:
            تصویر با تارشدگی کاهش یافته
        """
        kernel = np.zeros((kernel_size, kernel_size))
        
        if angle == 0:
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        else:
            # ایجاد کرنل مورب ساده
            for i in range(kernel_size):
                j = int(i * np.tan(np.radians(angle)))
                if 0 <= j < kernel_size:
                    kernel[i, j] = 1
        
        kernel = kernel / kernel.sum()
        
        # استفاده از فیلتر وینر برای دکانولوشن
        return ImageRestorer.wiener_filter(img, kernel_size, K=0.001)
    
    @staticmethod
    def gaussian_blur_remove(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        رفع تارشدگی گاوسی با استفاده از فیلتر وینر
        
        Args:
            img: تصویر ورودی
            sigma: انحراف معیار کرنل گاوسی
        
        Returns:
            تصویر شارپ شده
        """
        size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = cv2.getGaussianKernel(size, sigma)
        kernel = kernel @ kernel.T
        
        return ImageRestorer.wiener_filter(img, kernel_size=size, K=0.005)
    
    @staticmethod
    def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        فیلتر دوطرفه (حفظ لبه‌ها)
        
        Args:
            img: تصویر ورودی
       
        d: قطر پنجره
            sigma_color: انحراف معیار در فضای رنگ
            sigma_space: انحراف معیار در فضای مختصات
        
        Returns:
            تصویر فیلتر شده
        """
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    @staticmethod
    def median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        فیلتر میانه برای حذف نویز نمک و فلفل
        
        Args:
            img: تصویر ورودی
            kernel_size: اندازه کرنل
        
        Returns:
            تصویر بدون نویز
        """
        return cv2.medianBlur(img, kernel_size)
    
    @staticmethod
    def full_restore_pipeline(img: np.ndarray) -> np.ndarray:
        """
        خط لوله کامل بازسازی تصویر
        
        Args:
            img: تصویر ورودی
        
        Returns:
            تصویر بازسازی شده
        """
        # مرحله 1: حذف نویز با NLM
        img = ImageRestorer.non_local_means_denoise(img, h=8)
        
        # مرحله 2: رفع تارشدگی خفیف
        img = ImageRestorer.wiener_filter(img, kernel_size=3, K=0.005)
        
        # مرحله 3: شارپ‌سازی نهایی
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel_sharpen)
        
        return img
    
    @staticmethod
    def remove_noise_adaptive(img: np.ndarray) -> np.ndarray:
        """
        حذف نویز تطبیقی بر اساس نوع نویز تشخیص داده شده
        
        Args:
            img: تصویر ورودی
        
        Returns:
            تصویر بدون نویز
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # تخمین سطح نویز
        noise_std = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        
        if noise_std < 5:
            # نویز کم - فیلتر خفیف
            return ImageRestorer.bilateral_filter(img, d=5, sigma_color=25, sigma_space=25)
        elif noise_std < 15:
            # نویز متوسط - NLM معمولی
            return ImageRestorer.non_local_means_denoise(img, h=10)
        else:
            # نویز زیاد - NLM قوی
            return ImageRestorer.non_local_means_denoise(img, h=20)
            
            
            