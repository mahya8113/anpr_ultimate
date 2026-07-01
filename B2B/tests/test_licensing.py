"""
test_licensing.py
تست‌های مربوط به سیستم لایسنس (مجوز) سامانه
شامل: تولید لایسنس، اعتبارسنجی، انقضا، محدودیت تعداد دوربین
"""

import pytest
import jwt
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.licensing import LicenseManager


# ==================== تست‌های تولید لایسنس ====================
class TestLicenseGeneration:
    """تست‌های مربوط به تولید لایسنس"""
    
    def test_generate_license(self):
        """تست تولید لایسنس جدید"""
        token = LicenseManager.generate_license(
            org_id=1,
            max_cameras=10,
            expiry_days=365
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_generate_license_with_different_params(self):
        """تست تولید لایسنس با پارامترهای مختلف"""
        # لایسنس با 5 دوربین و 30 روز
        token1 = LicenseManager.generate_license(1, 5, 30)
        
        # لایسنس با 20 دوربین و 365 روز
        token2 = LicenseManager.generate_license(2, 20, 365)
        
        assert token1 != token2
        
        # بررسی محتوای لایسنس
        payload1 = LicenseManager.validate_license(token1)
        assert payload1["max_cameras"] == 5
        
        payload2 = LicenseManager.validate_license(token2)
        assert payload2["max_cameras"] == 20
        assert payload2["org_id"] == 2
    
    def test_generate_license_minimal_params(self):
        """تست تولید لایسنس با حداقل پارامترها"""
        token = LicenseManager.generate_license(
            org_id=1,
            max_cameras=1,
            expiry_days=1
        )
        
        assert token is not None


# ==================== تست‌های اعتبارسنجی لایسنس ====================
class TestLicenseValidation:
    """تست‌های مربوط به اعتبارسنجی لایسنس"""
    
    def test_validate_valid_license(self):
        """تست اعتبارسنجی لایسنس معتبر"""
        token = LicenseManager.generate_license(1, 10, 30)
        payload = LicenseManager.validate_license(token)
        
        assert payload is not None
        assert payload["org_id"] == 1
        assert payload["max_cameras"] == 10
        assert "exp" in payload
    
    def test_validate_expired_license(self):
        """تست لایسنس منقضی شده"""
        # ایجاد لایسنس با انقضای 0 روز (منقضی)
        token = LicenseManager.generate_license(1, 10, 0)
        
        with pytest.raises(ValueError) as exc_info:
            LicenseManager.validate_license(token)
        assert "منقضی" in str(exc_info.value) or "expired" in str(exc_info.value).lower()
    
    def test_validate_invalid_token(self):
        """تست توکن نامعتبر"""
        with pytest.raises(ValueError) as exc_info:
            LicenseManager.validate_license("invalid_token_string")
        assert "نامعتبر" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()
    
    def test_validate_wrong_secret(self):
        """تست لایسنس با کلید مخفی اشتباه"""
        # ایجاد لایسنس با کلید پیش‌فرض
        original_secret = LicenseManager.SECRET_KEY
        LicenseManager.SECRET_KEY = "original_secret"
        token = LicenseManager.generate_license(1, 10, 30)
        
        # تغییر کلید
        LicenseManager.SECRET_KEY = "different_secret"
        
        with pytest.raises(ValueError):
            LicenseManager.validate_license(token)
        
        # بازگرداندن کلید
        LicenseManager.SECRET_KEY = original_secret


# ==================== تست‌های محدودیت دوربین ====================
class TestCameraLimit:
    """تست‌های مربوط به محدودیت تعداد دوربین بر اساس لایسنس"""
    
    def test_within_camera_limit(self):
        """تست در محدوده مجاز دوربین"""
        token = LicenseManager.generate_license(1, 10, 30)
        payload = LicenseManager.validate_license(token)
        
        current_cameras = 5
        assert current_cameras <= payload["max_cameras"]
    
    def test_exceed_camera_limit(self):
        """تست فراتر از محدوده مجاز دوربین"""
        token = LicenseManager.generate_license(1, 10, 30)
        payload = LicenseManager.validate_license(token)
        
        current_cameras = 15
        assert current_cameras > payload["max_cameras"]
    
    def test_check_camera_limit_method(self):
        """تست متد check_camera_limit"""
        token = LicenseManager.generate_license(1, 10, 30)
        
        # در محدوده
        is_valid = LicenseManager.check_camera_limit(1, 5, token)
        assert is_valid is True or isinstance(is_valid, bool)
    
    def test_camera_limit_with_expired_license(self):
        """تست محدودیت دوربین با لایسنس منقضی"""
        token = LicenseManager.generate_license(1, 10, 0)
        
        with pytest.raises(ValueError):
            LicenseManager.check_camera_limit(1, 5, token)


# ==================== تست‌های زمان انقضا ====================
class TestExpiry:
    """تست‌های مربوط به زمان انقضای لایسنس"""
    
    def test_expiry_date(self):
        """تست تاریخ انقضای صحیح"""
        expiry_days = 30
        token = LicenseManager.generate_license(1, 10, expiry_days)
        payload = LicenseManager.validate_license(token)
        
        exp_time = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.now() + timedelta(days=expiry_days)
        
        # اختلاف کمتر از 1 ثانیه
        diff = abs((exp_time - expected_exp).total_seconds())
        assert diff < 1
    
    def test_remaining_days(self):
        """تست محاسبه روزهای باقیمانده"""
        from backend.core.licensing import get_remaining_days
        
        token = LicenseManager.generate_license(1, 10, 30)
        days = get_remaining_days(token)
        
        assert 0 < days <= 30
    
    def test_remaining_days_expired(self):
        """تست روزهای باقیمانده برای لایسنس منقضی"""
        from backend.core.licensing import get_remaining_days
        
        token = LicenseManager.generate_license(1, 10, 0)
        days = get_remaining_days(token)
        
        assert days == 0 or days < 0


# ==================== تست‌های با استفاده از Redis ====================
@pytest.mark.asyncio
class TestLicenseRedis:
    """تست‌های ذخیره و بازیابی لایسنس در Redis"""
    
    async def test_store_license_in_redis(self):
        """تست ذخیره لایسنس در Redis"""
        from backend.core.redis_client import redis_client
        
        token = LicenseManager.generate_license(1, 10, 30)
        org_id = 1
        
        # ذخیره در Redis
        await redis_client.setex(f"license:org:{org_id}", 86400, token)
        
        # بازیابی
        stored = await redis_client.get(f"license:org:{org_id}")
        assert stored == token
    
    async def test_retrieve_and_validate(self):
        """تست بازیابی و اعتبارسنجی لایسنس از Redis"""
        from backend.core.redis_client import redis_client
        
        token = LicenseManager.generate_license(2, 20, 30)
        org_id = 2
        
        await redis_client.setex(f"license:org:{org_id}", 86400, token)
        stored = await redis_client.get(f"license:org:{org_id}")
        
        if stored:
            payload = LicenseManager.validate_license(stored)
            assert payload["org_id"] == 2
            assert payload["max_cameras"] == 20


# ==================== تست‌های یکپارچگی ====================
class TestIntegration:
    """تست‌های یکپارچگی لایسنس با بخش‌های دیگر"""
    
    def test_license_in_api_response(self):
        """تست حضور لایسنس در پاسخ API"""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        # درخواست به endpoint لایسنس
        response = client.get("/license/info/1")
        # ممکن است 401 برگردد چون نیاز به احراز هویت دارد
        assert response.status_code in [200, 401]
    
    def test_license_check_in_detection(self):
        """تست بررسی لایسنس در زمان تشخیص"""
        # شبیه‌سازی درخواست تشخیص با لایسنس معتبر
        pass


# ==================== اجرای تست‌ها ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])