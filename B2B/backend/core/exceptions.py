"""
exceptions.py - کلاس‌های خطای سفارشی سامانه
"""

from typing import Optional, Any


class ANPRException(Exception):
    """کلاس پایه خطاهای سامانه"""
    
    def __init__(self, message: str, code: str = "ANPR_ERROR", status_code: int = 500, details: Optional[Any] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().init(self.message)


class ValidationError(ANPRException):
    """خطای اعتبارسنجی داده‌ها"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().init(message, code="VALIDATION_ERROR", status_code=422, details=details)


class AuthenticationError(ANPRException):
    """خطای احراز هویت"""
    
    def __init__(self, message: str = "نام کاربری یا رمز عبور اشتباه است"):
        super().init(message, code="AUTH_ERROR", status_code=401)


class AuthorizationError(ANPRException):
    """خطای دسترسی"""
    
    def __init__(self, message: str = "شما دسترسی به این بخش ندارید"):
        super().init(message, code="FORBIDDEN", status_code=403)


class NotFoundError(ANPRException):
    """خطای یافت نشدن"""
    
    def __init__(self, message: str = "مورد درخواستی یافت نشد"):
        super().init(message, code="NOT_FOUND", status_code=404)


class LicenseError(ANPRException):
    """خطای لایسنس"""
    
    def __init__(self, message: str = "لایسنس نامعتبر یا منقضی شده است"):
        super().init(message, code="LICENSE_ERROR", status_code=403)


class RateLimitError(ANPRException):
    """خطای نرخ محدودیت"""
    
    def __init__(self, message: str = "تعداد درخواست‌های شما بیش از حد مجاز است", wait_seconds: int = 60):
        super().init(message, code="RATE_LIMIT", status_code=429)
        self.wait_seconds = wait_seconds


class ModelLoadError(ANPRException):
    """خطا در بارگذاری مدل"""
    
    def __init__(self, model_name: str, message: str = "خطا در بارگذاری مدل"):
        super().init(f"{message}: {model_name}", code="MODEL_ERROR", status_code=500)


class DatabaseError(ANPRException):
    """خطای دیتابیس"""
    
    def __init__(self, message: str):
        super().init(message, code="DATABASE_ERROR", status_code=500)


class CameraError(ANPRException):
    """خطای دوربین"""
    
    def __init__(self, message: str):
        super().init(message, code="CAMERA_ERROR", status_code=400)


class DetectionError(ANPRException):
    """خطای تشخیص پلاک"""
    
    def __init__(self, message: str):
        super().init(message, code="DETECTION_ERROR", status_code=500)


class OCRError(ANPRException):
    """خطای OCR"""
    
    def __init__(self, message: str):
        super().init(message, code="OCR_ERROR", status_code=500)


# توابع کمکی برای خطاها
def handle_exception(e: Exception) -> dict:
    """تبدیل خطا به پاسخ مناسب"""
    if isinstance(e, ANPRException):
        return {
            "success": False,
            "error": e.code,
            "message": e.message,
            "status_code": e.status_code,
            "details": e.details
        }
    
    return {
        "success": False,
        "error": "INTERNAL_ERROR",
        "message": "خطای داخلی سرور",
        "status_code": 500,
        "details": str(e)
    }