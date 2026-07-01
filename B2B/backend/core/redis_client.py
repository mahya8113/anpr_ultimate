"""
redis_client.py - مدیریت اتصال به Redis
"""

import redis.asyncio as redis
from typing import Optional, Any
import json
import logging
from .config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None


async def init_redis():
    """مقداردهی اولیه اتصال Redis"""
    global _redis_client
    
    try:
        _redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        await _redis_client.ping()
        logger.info("✅ اتصال به Redis برقرار شد")
        return _redis_client
    except Exception as e:
        logger.error(f"❌ خطا در اتصال به Redis: {e}")
        raise


async def close_redis():
    """بستن اتصال Redis"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        logger.info("✅ اتصال Redis بسته شد")


def get_redis() -> redis.Redis:
    """دریافت کلاینت Redis"""
    if _redis_client is None:
        raise Exception("Redis not initialized. Call init_redis() first.")
    return _redis_client


class RedisClient:
    """کلاس wrapper برای عملیات Redis"""
    
    @staticmethod
    async def set(key: str, value: Any, ttl: Optional[int] = None):
        """ذخیره مقدار در Redis"""
        client = get_redis()
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        if ttl:
            await client.setex(key, ttl, value)
        else:
            await client.set(key, value)
    
    @staticmethod
    async def get(key: str) -> Optional[Any]:
        """دریافت مقدار از Redis"""
        client = get_redis()
        value = await client.get(key)
        if value and (value.startswith('{') or value.startswith('[')):
            try:
                return json.loads(value)
            except:
                return value
        return value
    
    @staticmethod
    async def delete(key: str):
        """حذف کلید از Redis"""
        client = get_redis()
        await client.delete(key)
    
    @staticmethod
    async def exists(key: str) -> bool:
        """بررسی وجود کلید در Redis"""
        client = get_redis()
        return await client.exists(key) > 0
    
    @staticmethod
    async def expire(key: str, ttl: int):
        """تنظیم زمان انقضا برای کلید"""
        client = get_redis()
        await client.expire(key, ttl)
    
    @staticmethod
    async def incr(key: str) -> int:
        """افزایش مقدار عددی"""
        client = get_redis()
        return await client.incr(key)


# کلاینت مستقیم برای استفاده در سایر ماژول‌ها
redis_client = RedisClient()
