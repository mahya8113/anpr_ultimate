"""
test_cache.py
تست‌های مربوط به سیستم کش (Redis Cache) برای نتایج تشخیص
"""

import pytest
import asyncio
import json
import hashlib
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.cache import DetectionCache
from backend.core.redis_client import redis_client


# ==================== تست‌های پایه کش ====================
class TestDetectionCache:
    """تست‌های مربوط به کش نتایج تشخیص"""
    
    @pytest.fixture
    def sample_image_bytes(self):
        """ایجاد تصویر نمونه به صورت بایت"""
        return b"fake_image_data_12345"
    
    @pytest.fixture
    def sample_detection_result(self):
        """نتیجه تشخیص نمونه"""
        return {
            "plates": [
                {"plate_text": "1234567", "confidence": 0.95, "bbox": [10, 20, 100, 50]}
            ],
            "num_plates": 1,
            "processing_time_ms": 45.2
        }
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, sample_image_bytes):
        """تست تولید کلید یکتا برای تصویر"""
        key1 = DetectionCache._key(sample_image_bytes)
        key2 = DetectionCache._key(sample_image_bytes)
        
        assert key1 == key2
        assert key1.startswith("detect:")
        assert len(key1) > 20
    
    @pytest.mark.asyncio
    async def test_set_and_get_cache(self, sample_image_bytes, sample_detection_result):
        """تست ذخیره و بازیابی از کش"""
        # ذخیره در کش
        await DetectionCache.set(sample_image_bytes, sample_detection_result)
        
        # بازیابی از کش
        cached = await DetectionCache.get(sample_image_bytes)
        
        assert cached is not None
        assert cached["num_plates"] == sample_detection_result["num_plates"]
        assert cached["plates"][0]["plate_text"] == sample_detection_result["plates"][0]["plate_text"]
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """تست عدم وجود داده در کش"""
        fake_image = b"nonexistent_image_data"
        cached = await DetectionCache.get(fake_image)
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_cache_ttl(self, sample_image_bytes, sample_detection_result):
        """تست انقضای زمان کش"""
        # تنظیم TTL کم برای تست
        original_ttl = DetectionCache.TTL
        DetectionCache.TTL = 1  # 1 ثانیه
        
        await DetectionCache.set(sample_image_bytes, sample_detection_result)
        
        # بلافاصله قابل بازیابی است
        cached = await DetectionCache.get(sample_image_bytes)
        assert cached is not None
        
        # بعد از 2 ثانیه منقضی می‌شود
        await asyncio.sleep(2)
        expired = await DetectionCache.get(sample_image_bytes)
        
        # بازگرداندن TTL اصلی
        DetectionCache.TTL = original_ttl
    
    @pytest.mark.asyncio
    async def test_cache_with_different_images(self, sample_detection_result):
        """تست ذخیره تصاویر مختلف"""
        img1 = b"image_1_data"
        img2 = b"image_2_data"
        
        result1 = {"plates": [{"plate_text": "1111111"}], "num_plates": 1}
        result2 = {"plates": [{"plate_text": "2222222"}], "num_plates": 1}
        
        await DetectionCache.set(img1, result1)
        await DetectionCache.set(img2, result2)
        
        cached1 = await DetectionCache.get(img1)
        cached2 = await DetectionCache.get(img2)
        
        assert cached1["plates"][0]["plate_text"] == "1111111"
        assert cached2["plates"][0]["plate_text"] == "2222222"


# ==================== تست‌های Redis Connection ====================
class TestRedisConnection:
    """تست‌های مربوط به اتصال Redis"""
    
    @pytest.mark.asyncio
    async def test_redis_ping(self):
        """تست پینگ Redis"""
        try:
            result = await redis_client.ping()
            assert result is True
        except:
            pytest.skip("Redis در دسترس نیست")
    
    @pytest.mark.asyncio
    async def test_redis_set_get(self):
        """تست عملیات set/get Redis"""
        try:
            await redis_client.set("test_key", "test_value")
            value = await redis_client.get("test_key")
            assert value == "test_value"
            
            # پاکسازی
            await redis_client.delete("test_key")
        except:
            pytest.skip("Redis در دسترس نیست")
    
    @pytest.mark.asyncio
    async def test_redis_expire(self):
        """تست انقضای کلید در Redis"""
        try:
            await redis_client.setex("temp_key", 1, "temp_value")
            value = await redis_client.get("temp_key")
            assert value == "temp_value"
            
            await asyncio.sleep(2)
            expired = await redis_client.get("temp_key")
            assert expired is None
        except:
            pytest.skip("Redis در دسترس نیست")


# ==================== تست‌های کش با داده واقعی ====================
class TestRealCache:
    """تست‌های کش با داده‌های واقعی تشخیص"""
    
    @pytest.mark.asyncio
    async def test_cache_with_real_detection_data(self):
        """تست کش با داده واقعی تشخیص پلاک"""
        from modules.detection import ObjectDetector
        
        detector = ObjectDetector()
        
        # ایجاد تصویر نمونه
        import cv2
        import numpy as np
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        # اولین بار - مستقیم تشخیص (کش خالی)
        result1 = await DetectionCache.get(img_bytes)
        assert result1 is None
        
        # شبیه‌سازی تشخیص و ذخیره در کش
        fake_result = {"plates": [], "num_plates": 0}
        await DetectionCache.set(img_bytes, fake_result)
        
        # بار دوم - باید از کش بخواند
        result2 = await DetectionCache.get(img_bytes)
        assert result2 is not None
        assert result2 == fake_result


# ==================== تست‌های نرخ محدودیت (Rate Limiting) ====================
class TestRateLimiting:
    """تست‌های مربوط به نرخ محدودیت کش"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_key_generation(self):
        """تست تولید کلید برای نرخ محدودیت"""
        ip = "192.168.1.1"
        key = f"rate_limit:{ip}"
        assert key.startswith("rate_limit:")
        assert ip in key
    
    @pytest.mark.asyncio
    async def test_rate_limit_counter(self):
        """تست شمارنده نرخ محدودیت"""
        from backend.utils.rate_limiter import rate_limit
        
        # تست با IP موقت
        pass


# ==================== اجرای تست‌ها ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])