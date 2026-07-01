"""
auth.py - مسیرهای احراز هویت (ورود، ثبت نام، خروج، توکن)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import jwt
import os

from core.database import get_db
from core.security import hash_password, verify_password
from core.redis_client import redis_client

router = APIRouter(prefix="/auth", tags=["احراز هویت"])

# تنظیمات JWT
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

security = HTTPBearer()


# ==================== مدل‌های Pydantic ====================
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    org_name: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# ==================== توابع کمکی ====================
def create_tokens(user_id: int, role: str):
    """ایجاد access و refresh token"""
    now = datetime.utcnow()
    access_exp = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_exp = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_payload = {
        "sub": str(user_id),
        "role": role,
        "exp": access_exp,
        "type": "access"
    }
    refresh_payload = {
        "sub": str(user_id),
        "exp": refresh_exp,
        "type": "refresh"
    }
    access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)
    return access_token, refresh_token


def decode_token(token: str, expected_type: str = "access"):
    """اعتبارسنجی و دیکد توکن"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != expected_type:
            raise HTTPException(401, "نوع توکن نامعتبر است")
        # بررسی بلیک‌لیست
        is_blacklisted = await redis_client.get(f"bl_{token}")
        if is_blacklisted:
            raise HTTPException(401, "توکن باطل شده است")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "توکن منقضی شده است")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "توکن نامعتبر است")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """دریافت کاربر جاری از توکن"""
    token = credentials.credentials
    payload = decode_token(token, "access")
    user_id = int(payload["sub"])
    # در صورت نیاز اطلاعات کامل کاربر از دیتابیس قابل دریافت است
    return {"id": user_id, "role": payload.get("role")}


# ==================== مسیرها ====================
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: AsyncSession = Depends(get_db)):
    """ثبت نام کاربر جدید و ایجاد سازمان"""
    from models import Organization, User
    
    # بررسی تکراری نبودن ایمیل
    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "این ایمیل قبلاً ثبت شده است")
    
    # ایجاد سازمان (اگر وجود نداشته باشد)
    org = await db.execute(select(Organization).where(Organization.name == user_data.org_name))
    org = org.scalar_one_or_none()
    if not org:
        org = Organization(name=user_data.org_name, tier="standard", max_cameras=5, quota_limit=5000)
        db.add(org)
        await db.flush()
    
    # ایجاد کاربر
    hashed = hash_password(user_data.password)
    new_user = User(
        org_id=org.id,
        email=user_data.email,
        password_hash=hashed,
        full_name=user_data.full_name,
        role="admin"  # اولین کاربر سازمان نقش ادمین دارد
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # تولید توکن‌ها
    access, refresh = create_tokens(new_user.id, new_user.role)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    """ورود به سامانه و دریافت توکن"""
    from models import User
    
    # یافتن کاربر
    result = await db.execute(select(User).where(User.email == user_data.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(401, "ایمیل یا رمز عبور اشتباه است")
    
    if not user.is_active:
        raise HTTPException(403, "حساب کاربری غیرفعال است")
    
    # به‌روزرسانی آخرین ورود
    user.last_login = datetime.now()
    await db.commit()
    
    # تولید توکن‌ها
    access, refresh = create_tokens(user.id, user.role)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_data: RefreshRequest):
    """دریافت access token جدید با استفاده از refresh token"""
    payload = decode_token(refresh_data.refresh_token, "refresh")
    user_id = int(payload["sub"])
    
    # بازیابی نقش کاربر (از دیتابیس یا کش)
    # در اینجا فرض می‌کنیم نقش کاربر در payload موجود نیست، از دیتابیس می‌خوانیم
    from models import User
    async with AsyncSession() as db:
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(404, "کاربر یافت نشد")
        role = user.role
    
    access, _ = create_tokens(user_id, role)
    return TokenResponse(access_token=access, refresh_token=refresh_data.refresh_token)


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """خروج از سامانه (باطل کردن توکن فعلی)"""
    token = credentials.credentials
    payload = decode_token(token, "access")
    exp = payload.get("exp")
    if exp:
        ttl = max(0, exp - int(datetime.utcnow().timestamp()))
        await redis_client.setex(f"bl_{token}", ttl, "1")
    return {"message": "خروج با موفقیت انجام شد"}


@router.get("/me")
async def get_me(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """دریافت اطلاعات کاربر جاری"""
    from models import User
    
    user = await db.get(User, current_user["id"])
    if not user:
        raise HTTPException(404, "کاربر یافت نشد")
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "org_id": user.org_id,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "last_login": user.last_login
    }
    