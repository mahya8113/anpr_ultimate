"""
validators.py - توابع اعتبارسنجی داده‌ها
اعتبارسنجی: ایمیل، شماره تلفن، آدرس IP، URL، فرمت پلاک، و غیره
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse
import ipaddress


class Validators:
    """کلاس توابع اعتبارسنجی"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        اعتبارسنجی ایمیل
        
        Args:
            email: آدرس ایمیل
        
        Returns:
            True اگر معتبر باشد
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        اعتبارسنجی شماره تلفن ایران
        
        Args:
            phone: شماره تلفن
        
        Returns:
            True اگر معتبر باشد
        """
        # حذف فاصله و خط تیره
        phone = re.sub(r'[\s\-]', '', phone)
        
        # الگوهای معتبر شماره تلفن ایران
        patterns = [
            r'^09[0-9]{9}$',           # 09123456789
            r'^\+989[0-9]{9}$',        # +989123456789
            r'^00989[0-9]{9}$',        # 00989123456789
            r'^0[1-8][0-9]{9}$',       # 02112345678
        ]
        
        return any(re.match(pattern, phone) for pattern in patterns)
    
    @staticmethod
    def validate_ip(ip: str) -> bool:
        """
        اعتبارسنجی آدرس IP (IPv4 و IPv6)
        
        Args:
            ip: آدرس IP
        
        Returns:
            True اگر معتبر باشد
        """
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_url(url: str, allowed_schemes: list = None) -> bool:
        """
        اعتبارسنجی URL
        
        Args:
            url: آدرس URL
            allowed_schemes: لیست پروتکل‌های مجاز
        
        Returns:
            True اگر معتبر باشد
        """
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https', 'rtsp', 'rtmp']
        
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in allowed_schemes and
                parsed.netloc and
                len(url) < 2048
            )
        except Exception:
            return False
    
    @staticmethod
    def validate_persian_plate(plate: str) -> Tuple[bool, Optional[str]]:
        """
        اعتبارسنجی پلاک ایرانی
        
        Args:
            plate: شماره پلاک
        
        Returns:
            (معتبر بودن, پیام خطا)
        """
        # حذف فاصله‌ها
        plate = re.sub(r'\s+', '', plate)
        
        # الگوی پلاک ایران: 2 حرف + 3 رقم + 2 رقم + 1 حرف
        # مثال: ۱۲ب۳۴۵۶۷
        # یا: 1234567
        
        if len(plate) == 7 and plate.isdigit():
            return True, None
        
        # الگوی کامل فارسی
        persian_pattern = r'^[\u0600-\u06FF]{1,2}[۰-۹]{3}[۰-۹]{2}[\u0600-\u06FF]{1}$'
        if re.match(persian_pattern, plate):
            return True, None
        
        # الگوی کامل انگلیسی
        english_pattern = r'^[A-Z]{1,2}[0-9]{3}[0-9]{2}[A-Z]{1}$'
        if re.match(english_pattern, plate):
            return True, None
        
        return False, "فرمت پلاک نامعتبر است"
    
    @staticmethod
    def validate_stream_url(url: str) -> Tuple[bool, str]:
        """
        اعتبارسنجی آدرس استریم دوربین
        
        Args:
            url: آدرس دوربین
        
        Returns:
            (معتبر بودن, پیام)
        """
        parsed = urlparse(url)
        
        if parsed.scheme == 'rtsp':
            if not parsed.netloc:
                return False, "آدرس RTSP نامعتبر است"
            return True, "آدرس RTSP معتبر است"
        
        elif parsed.scheme in ('http', 'https'):
            if 'mjpeg' not in url.lower() and 'stream' not in url.lower():
                return False, "آدرس HTTP باید شامل mjpeg یا stream باشد"
            return True, "آدرس HTTP معتبر است"
        
        elif parsed.scheme == 'usb':
            try:
                camera_id = int(parsed.netloc) if parsed.netloc else 0
                if camera_id < 0 or camera_id > 10:
                    return False, "شناسه دوربین USB باید بین 0 تا 10 باشد"
                return True, f"دوربین USB {camera_id} معتبر است"
            except ValueError:
                return False, "شناسه دوربین USB نامعتبر است"
        
        elif parsed.scheme == 'v4l2':
            device = parsed.netloc or '/dev/video0'
            if not device.startswith('/dev/video'):
                return False, "مسیر دستگاه V4L2 نامعتبر است"
            return True, f"دستگاه V4L2 {device} معتبر است"
        
        else:
            return False, f"پروتکل {parsed.scheme} پشتیبانی نمی‌شود"
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        اعتبارسنجی رمز عبور
        
        Args:
            password: رمز عبور
        
        Returns:
            (معتبر بودن, پیام خطا)
        """
        if len(password) < 8:
            return False, "رمز عبور باید حداقل 8 کاراکتر باشد"
        
        if not any(c.isupper() for c in password):
            return False, "رمز عبور باید حداقل یک حرف بزرگ داشته باشد"
        
        if not any(c.islower() for c in password):
            return False, "رمز عبور باید حداقل یک حرف کوچک داشته باشد"
        
        if not any(c.isdigit() for c in password):
            return False, "رمز عبور باید حداقل یک عدد داشته باشد"
        
        if not any(c in '!@#$%^&*()_+-=[]{};:,.<>?/' for c in password):
            return False, "رمز عبور باید حداقل یک کاراکتر خاص داشته باشد"
        
        return True, "رمز عبور معتبر است"
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
        """
        اعتبارسنجی پسوند فایل
        
        Args:
            filename: نام فایل
            allowed_extensions: لیست پسوندهای مجاز
        
        Returns:
            True اگر معتبر باشد
        """
        ext = filename.rsplit('.', 1)[-1].lower()
        return ext in allowed_extensions
    
    @staticmethod
    def validate_image_file(filename: str) -> bool:
        """اعتبارسنجی فایل تصویر"""
        allowed = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        return Validators.validate_file_extension(filename, allowed)
    
    @staticmethod
    def validate_video_file(filename: str) -> bool:
        """اعتبارسنجی فایل ویدئو"""
        allowed = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        return Validators.validate_file_extension(filename, allowed)
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        پاکسازی ورودی برای جلوگیری از XSS و SQL Injection
        
        Args:
            text: متن ورودی
        
        Returns:
            متن پاکسازی شده
        """
        if not text:
            return ""
        
        # حذف تگ‌های HTML
        text = re.sub(r'<[^>]*>', '', text)
        
        # حذف کاراکترهای خاص
        text = re.sub(r'[;`\'\"\\]', '', text)
        
        return text.strip()


# ==================== Pydantic Validators ====================

from pydantic import validator, BaseModel
from typing import Optional


class PlateDetectionRequest(BaseModel):
    """مدل اعتبارسنجی درخواست تشخیص پلاک"""
    
    image_base64: str
    camera_id: Optional[int] = None
    
    @validator('image_base64')
    def validate_base64(cls, v):
        if not v or len(v) < 10:
            raise ValueError('تصویر نامعتبر است')
        return v


class CameraCreateRequest(BaseModel):
    """مدل اعتبارسنجی ایجاد دوربین"""
    
    name: str
    stream_url: str
    stream_type: str
    location: Optional[str] = None
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 2 or len(v) > 100:
            raise ValueError('نام دوربین باید بین 2 تا 100 کاراکتر باشد')
        return Validators.sanitize_input(v)
    
    @validator('stream_url')
    def validate_stream_url(cls, v):
        valid, message = Validators.validate_stream_url(v)
        if not valid:
            raise ValueError(message)
        return v
    
    @validator('stream_type')
    def validate_stream_type(cls, v):
        allowed = ['rtsp', 'http', 'usb', 'v4l2', 'onvif']
        if v not in allowed:
            raise ValueError(f'نوع استریم باید یکی از {allowed} باشد')
        return v