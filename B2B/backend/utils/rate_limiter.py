"""
rate_limiter.py - مدیریت محدودیت نرخ درخواست‌ها (Rate Limiting)
با استفاده از Redis و الگوریتم Sliding Window
"""

import time
import asyncio
from typing import Optional, Tuple
from fastapi import HTTPException, Request
from core.redis_client import redis_client
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """محدودیت نرخ درخواست‌ها با الگوریتم Sliding Window"""
    
    def init(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        block_duration: int = 300
    ):
        """
        Args:
            requests_per_window: تعداد مجاز درخواست در بازه زمانی
            window_seconds: طول بازه زمانی به ثانیه
            block_duration: مدت زمان بلاک در صورت نقض محدودیت (ثانیه)
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.block_duration = block_duration
    
    async def check_rate_limit(self, key: str) -> Tuple[bool, int]:
        """
        بررسی محدودیت نرخ برای یک کلید خاص
        
        Args:
            key: کلید شناسایی (معمولاً IP یا User ID)
        
        Returns:
            (مجاز بودن, زمان باقیمانده)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # بررسی بلاک بودن
        block_key = f"blocked:{key}"
        is_blocked = await redis_client.get(block_key)
        if is_blocked:
            ttl = await redis_client.ttl(block_key)
            return False, ttl if ttl > 0 else 0
        
        # ذخیره زمان درخواست در Redis
        redis_key = f"rate_limit:{key}"
        
        # حذف درخواست‌های قدیمی
        await redis_client.zremrangebyscore(redis_key, 0, window_start)
        
        # شمارش درخواست‌های موجود
        current_count = await redis_client.zcard(redis_key)
        
        if current_count >= self.requests_per_window:
            # بلاک کردن کاربر
            await redis_client.setex(block_key, self.block_duration, "1")
            await redis_client.delete(redis_key)
            logger.warning(f"نرخ محدودیت فعال شد برای کلید {key}")
            return False, self.block_duration
        
        # ثبت درخواست جدید
        await redis_client.zadd(redis_key, {str(now): now})
        await redis_client.expire(redis_key, self.window_seconds)
        
        remaining = self.requests_per_window - (current_count + 1)
        return True, remaining
    
    async def call(self, request: Request) -> None:
        """استفاده به عنوان middleware FastAPI"""
        client_ip = request.client.host
        api_key = request.headers.get("X-API-Key")
        
        # استفاده از API Key اگر موجود باشد، در غیر این صورت IP
        identifier = api_key if api_key else client_ip
        key = f"rate:{identifier}"
        
        allowed, remaining_or_wait = await self.check_rate_limit(key)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "تعداد درخواست‌های شما بیش از حد مجاز است",
                    "wait_seconds": remaining_or_wait,
                    "try_again_after": int(time.time() + remaining_or_wait)
                }
            )
        
        # اضافه کردن هدرها به پاسخ
        request.state.rate_limit_remaining = remaining_or_wait


class WebSocketRateLimiter:
    """محدودیت نرخ برای WebSocket"""
    
    def init(self, messages_per_minute: int = 60):
        self.messages_per_minute = messages_per_minute
    
    async def check_rate_limit(self, websocket, key: str) -> bool:
        """بررسی محدودیت برای WebSocket"""
        now = time.time()
        minute_ago = now - 60
        
        redis_key = f"ws_rate_limit:{key}"
        
        # حذف پیام‌های قدیمی
        await redis_client.zremrangebyscore(redis_key, 0, minute_ago)
        # شمارش پیام‌ها
        count = await redis_client.zcard(redis_key)
        
        if count >= self.messages_per_minute:
            await websocket.close(code=1008, reason="نرخ پیام بیش از حد مجاز است")
            return False
        
        # ثبت پیام جدید
        await redis_client.zadd(redis_key, {str(now): now})
        await redis_client.expire(redis_key, 60)
        
        return True


# ==================== توابع کمکی ====================

async def get_client_identifier(request: Request) -> str:
    """دریافت شناسه یکتای کلاینت"""
    # اولویت: API Key > User ID > IP Address
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    # اگر کاربر احراز هویت شده باشد
    if hasattr(request.state, "user_id"):
        return f"user:{request.state.user_id}"
    
    # در غیر این صورت از IP استفاده می‌شود
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0]
    else:
        client_ip = request.client.host
    
    return f"ip:{client_ip}"


async def reset_rate_limit(key: str) -> bool:
    """بازنشانی محدودیت نرخ برای یک کلید"""
    try:
        redis_key = f"rate_limit:{key}"
        block_key = f"blocked:{key}"
        
        await redis_client.delete(redis_key)
        await redis_client.delete(block_key)
        return True
    except Exception as e:
        logger.error(f"خطا در بازنشانی محدودیت: {e}")
        return False


class RateLimitMiddleware:
    """Middleware برای اعمال محدودیت نرخ در سطح برنامه"""
    
    def init(self, app, requests_per_minute: int = 60):
        self.app = app
        self.rate_limiter = RateLimiter(
            requests_per_window=requests_per_minute,
            window_seconds=60,
            block_duration=300
        )
    
    async def call(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # مسیرهای مستثنی
        excluded_paths = ["/health", "/metrics", "/docs", "/openapi.json"]
        if request.url.path in excluded_paths:
            await self.app(scope, receive, send)
            return
        
        identifier = await get_client_identifier(request)
        allowed, _ = await self.rate_limiter.check_rate_limit(identifier)
        
        if not allowed:
            response = HTTPException(429, "تعداد درخواست‌های شما بیش از حد مجاز است")
            await response(scope, receive, send)
            return
        
        await self.app(scope, receive, send)