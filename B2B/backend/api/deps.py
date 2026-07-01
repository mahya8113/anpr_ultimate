"""
deps.py - توابع وابستگی (Dependencies) برای احراز هویت، دسترسی به دیتابیس و ...
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
import os

from core.database import get_db
from core.redis_client import redis_client
from models import User

security = HTTPBearer()

# تنظیمات JWT
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-me")
ALGORITHM = "HS256"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    دریافت کاربر فعلی از توکن JWT
    Returns:
        شیء User از دیتابیس
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="نوع توکن نامعتبر است")
        
        user_id = int(payload.get("sub"))
        if not user_id:
            raise HTTPException(status_code=401, detail="توکن نامعتبر است")
        
        # بررسی بلیک‌لیست Redis
        blacklisted = await redis_client.get(f"bl_{token}")
        if blacklisted:
            raise HTTPException(status_code=401, detail="توکن باطل شده است")
        
        # دریافت کاربر از دیتابیس
        user = await db.get(User, user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="کاربر یافت نشد یا غیرفعال است")
        
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="توکن منقضی شده است")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="توکن نامعتبر است")


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """فقط ادمین‌ها (admin یا super_admin)"""
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="دسترسی محدود: فقط ادمین‌ها")
    return current_user


async def get_current_super_admin(current_user: User = Depends(get_current_user)) -> User:
    """فقط super_admin"""
    if current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="دسترسی محدود: فقط سوپر ادمین")
    return current_user


async def get_organization_admin(
    org_id: int,
    current_user: User = Depends(get_current_user)
) -> User:
    """
    ادمین سازمان (ادمین خود سازمان یا سوپر ادمین)
    """
    if current_user.role == "super_admin":
        return current_user
    if current_user.role == "admin" and current_user.org_id == org_id:
        return current_user
    raise HTTPException(status_code=403, detail="شما دسترسی به این سازمان ندارید")


async def verify_websocket_token(token: str) -> dict:
    """
    اعتبارسنجی توکن برای WebSocket (بدون وابستگی به FastAPI Depends)
    Returns:
        دیکشنری حاوی اطلاعات کاربر (id, role) یا None در صورت نامعتبری
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        
        user_id = int(payload.get("sub"))
        if not user_id:
            return None
        
        # بررسی بلیک‌لیست (اختیاری)
        blacklisted = await redis_client.get(f"bl_{token}")
        if blacklisted:
            return None
        
        return {"id": user_id, "role": payload.get("role", "operator")}
    except Exception:
        return None


async def get_db_session() -> AsyncSession:
    """ارائه جلسه دیتابیس (برای استفاده در جاهایی که Depends ندارد)"""
    async for session in get_db():
        return session


def get_cache_client():
    """ارائه کلاینت کش Redis"""
    return redis_client


# ==================== توابع کمکی برای نرخ محدودیت و IP ====================
async def get_client_ip(request) -> str:
    """دریافت IP واقعی کلاینت (با احتساب پروکسی)"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


# ==================== وابستگی برای نرخ محدودیت (در صورت نیاز) ====================

from utils.rate_limiter import RateLimiter

rate_limiter = RateLimiter(requests_per_window=100, window_seconds=60)


async def rate_limit(request):
    """اعمال محدودیت نرخ درخواست بر اساس IP کاربر"""
    client_ip = await get_client_ip(request)
    key = f"rate_limit:{client_ip}"
    allowed, remaining = await rate_limiter.check_rate_limit(key)
    if not allowed:
        raise HTTPException(status_code=429, detail="تعداد درخواست‌های شما بیش از حد مجاز است")
    return remaining