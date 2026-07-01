"""
cache.py - مدیریت کش نتایج تشخیص با Redis
"""

import hashlib
import json
from typing import Optional, Any
import logging
from .redis_client import redis_client

logger = logging.getLogger(__name__)


class DetectionCache:
    """کش نتایج تشخیص پلاک"""
    
    TTL = 3600  # 1 ساعت
    
    @classmethod
    def _key(cls, image_bytes: bytes) -> str:
        """تولید کلید یکتا برای تصویر"""
        return f"detect:{hashlib.md5(image_bytes).hexdigest()}"
    
    @classmethod
    async def get(cls, image_bytes: bytes) -> Optional[dict]:
        """دریافت نتیجه از کش"""
        key = cls._key(image_bytes)
        cached = await redis_client.get(key)
        
        if cached:
            logger.debug(f"کش hit برای کلید {key}")
            return cached
        logger.debug(f"کش miss برای کلید {key}")
        return None
    
    @classmethod
    async def set(cls, image_bytes: bytes, result: dict):
        """ذخیره نتیجه در کش"""
        key = cls._key(image_bytes)
        await redis_client.set(key, result, cls.TTL)
        logger.debug(f"نتیجه در کش ذخیره شد: {key}")
    
    @classmethod
    async def invalidate(cls, image_bytes: bytes):
        """حذف از کش"""
        key = cls._key(image_bytes)
        await redis_client.delete(key)
        logger.debug(f"کش برای کلید {key} حذف شد")


class RateLimitCache:
    """کش برای نرخ محدودیت"""
    
    @staticmethod
    async def check_rate_limit(key: str, limit: int, period: int) -> tuple:
        """
        بررسی محدودیت نرخ
        Returns: (allowed, remaining)
        """
        current = await redis_client.get(key)
        
        if current is None:
            await redis_client.set(key, 1, period)
            return True, limit - 1
        
        count = int(current) if isinstance(current, (int, str)) else 0
        
        if count >= limit:
            return False, 0
        
        await redis_client.incr(key)
        return True, limit - (count + 1)


class SessionCache:
    """کش جلسات کاربری"""
    
    TTL = 86400  # 24 ساعت
    
    @staticmethod
    async def set_session(session_id: str, data: dict):
        """ذخیره جلسه"""
        key = f"session:{session_id}"
        await redis_client.set(key, data, SessionCache.TTL)
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[dict]:
        """دریافت جلسه"""
        key = f"session:{session_id}"
        return await redis_client.get(key)
    
    @staticmethod
    async def delete_session(session_id: str):
        """حذف جلسه"""
        key = f"session:{session_id}"
        await redis_client.delete(key)
        