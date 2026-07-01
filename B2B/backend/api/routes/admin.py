"""
admin.py - مسیرهای مدیریت سازمانی (ادمین)
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from typing import List, Optional
from datetime import datetime
import logging

from core.database import get_db
from core.security import get_current_admin, hash_password
from models import Organization, User, Camera, Detection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["مدیریت سازمانی"])


# ==================== مدل‌های Pydantic ====================
from pydantic import BaseModel, EmailStr

class OrganizationCreate(BaseModel):
    name: str
    tier: str = "standard"
    max_cameras: int = 5
    quota_limit: int = 5000


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    tier: Optional[str] = None
    max_cameras: Optional[int] = None
    quota_limit: Optional[int] = None
    is_active: Optional[bool] = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "operator"


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class CameraCreate(BaseModel):
    name: str
    stream_url: str
    stream_type: str = "rtsp"
    location: Optional[str] = None


# ==================== 1. مدیریت سازمان‌ها ====================

@router.get("/organizations")
async def list_organizations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """لیست تمام سازمان‌ها (فقط ادمین اصلی)"""
    query = select(Organization)
    if search:
        query = query.where(Organization.name.ilike(f"%{search}%"))
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    orgs = result.scalars().all()
    
    response = []
    for org in orgs:
        user_count = await db.scalar(select(func.count(User.id)).where(User.org_id == org.id))
        camera_count = await db.scalar(select(func.count(Camera.id)).where(Camera.org_id == org.id))
        response.append({
            "id": org.id,
            "name": org.name,
            "tier": org.tier,
            "max_cameras": org.max_cameras,
            "quota_limit": org.quota_limit,
            "is_active": org.is_active,
            "created_at": org.created_at,
            "expires_at": org.expires_at,
            "user_count": user_count,
            "camera_count": camera_count
        })
    return response


@router.post("/organizations", status_code=status.HTTP_201_CREATED)
async def create_organization(
    org_data: OrganizationCreate,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """ایجاد سازمان جدید"""
    existing = await db.execute(select(Organization).where(Organization.name == org_data.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="سازمانی با این نام قبلاً ثبت شده است")
    
    new_org = Organization(**org_data.dict())
    db.add(new_org)
    await db.commit()
    await db.refresh(new_org)
    
    return {"message": "سازمان با موفقیت ایجاد شد", "organization_id": new_org.id}


@router.put("/organizations/{org_id}")
async def update_organization(
    org_id: int,
    org_data: OrganizationUpdate,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """به‌روزرسانی اطلاعات سازمان"""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="سازمان یافت نشد")
    
    for field, value in org_data.dict(exclude_unset=True).items():
        setattr(org, field, value)
    
    await db.commit()
    return {"message": "سازمان با موفقیت به‌روزرسانی شد"}
@router.delete("/organizations/{org_id}")
async def delete_organization(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """حذف سازمان و تمام داده‌های مرتبط"""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="سازمان یافت نشد")
    
    # حذف وابسته‌ها
    await db.execute(delete(User).where(User.org_id == org_id))
    await db.execute(delete(Camera).where(Camera.org_id == org_id))
    await db.execute(delete(Detection).where(Detection.org_id == org_id))
    await db.delete(org)
    await db.commit()
    
    return {"message": "سازمان حذف شد"}


# ==================== 2. مدیریت کاربران ====================

@router.get("/organizations/{org_id}/users")
async def list_users(
    org_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """لیست کاربران یک سازمان"""
    result = await db.execute(
        select(User).where(User.org_id == org_id).offset(skip).limit(limit)
    )
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "full_name": u.full_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at,
            "last_login": u.last_login
        } for u in users
    ]


@router.post("/organizations/{org_id}/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    org_id: int,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """ایجاد کاربر جدید در سازمان"""
    # بررسی وجود سازمان
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="سازمان یافت نشد")
    
    # بررسی تکراری نبودن ایمیل
    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="این ایمیل قبلاً ثبت شده است")
    
    new_user = User(
        org_id=org_id,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        full_name=user_data.full_name,
        role=user_data.role
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return {"message": "کاربر با موفقیت ایجاد شد", "user_id": new_user.id}


@router.put("/organizations/{org_id}/users/{user_id}")
async def update_user(
    org_id: int,
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """به‌روزرسانی اطلاعات کاربر"""
    user = await db.get(User, user_id)
    if not user or user.org_id != org_id:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    
    for field, value in user_data.dict(exclude_unset=True).items():
        setattr(user, field, value)
    
    await db.commit()
    return {"message": "کاربر با موفقیت به‌روزرسانی شد"}


@router.delete("/organizations/{org_id}/users/{user_id}")
async def delete_user(
    org_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """حذف کاربر"""
    user = await db.get(User, user_id)
    if not user or user.org_id != org_id:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد")
    
    await db.delete(user)
    await db.commit()
    return {"message": "کاربر حذف شد"}


# ==================== 3. مدیریت دوربین‌ها ====================

@router.get("/organizations/{org_id}/cameras")
async def list_cameras(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """لیست دوربین‌های یک سازمان"""
    result = await db.execute(select(Camera).where(Camera.org_id == org_id))
    cameras = result.scalars().all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "location": c.location,
            "stream_url": c.stream_url,
            "stream_type": c.stream_type,
            "is_active": c.is_active,
            "created_at": c.created_at
        } for c in cameras
    ]


@router.post("/organizations/{org_id}/cameras", status_code=status.HTTP_201_CREATED)
async def add_camera(
    org_id: int,
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """افزودن دوربین جدید به سازمان"""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="سازمان یافت نشد")
    
    current_count = await db.scalar(select(func.count(Camera.id)).where(Camera.org_id == org_id))
    if current_count >= org.max_cameras:
        raise HTTPException(
            status_code=400,
            detail=f"تعداد دوربین‌ها به حداکثر مجاز ({org.max_cameras}) رسیده است"
        )
    
    new_camera = Camera(**camera_data.dict(), org_id=org_id)
    db.add(new_camera)
    await db.commit()
    await db.refresh(new_camera)
    
    return {"message": "دوربین با موفقیت اضافه شد", "camera_id": new_camera.id}


@router.put("/organizations/{org_id}/cameras/{camera_id}")
async def update_camera(
    org_id: int,
    camera_id: int,
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """به‌روزرسانی اطلاعات دوربین"""
    camera = await db.get(Camera, camera_id)
    if not camera or camera.org_id != org_id:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    for field, value in camera_data.dict().items():
        setattr(camera, field, value)
    
    await db.commit()
    return {"message": "دوربین با موفقیت به‌روزرسانی شد"}


@router.delete("/organizations/{org_id}/cameras/{camera_id}")
async def delete_camera(
    org_id: int,
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """حذف دوربین"""
    camera = await db.get(Camera, camera_id)
    if not camera or camera.org_id != org_id:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    
    await db.delete(camera)
    await db.commit()
    return {"message": "دوربین حذف شد"}


# ==================== 4. آمار و داشبورد ====================

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """آمار کلی برای داشبورد ادمین"""
    total_orgs = await db.scalar(select(func.count(Organization.id)))
    total_users = await db.scalar(select(func.count(User.id)))
    total_cameras = await db.scalar(select(func.count(Camera.id)))
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    total_detections_today = await db.scalar(
        select(func.count(Detection.id)).where(Detection.created_at >= today_start)
    )
    
    return {
        "total_organizations": total_orgs or 0,
        "total_users": total_users or 0,
        "total_cameras": total_cameras or 0,
        "total_detections_today": total_detections_today or 0,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/dashboard/recent-detections")
async def get_recent_detections(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """دریافت آخرین تشخیص‌های پلاک"""
    result = await db.execute(
        select(Detection).order_by(Detection.created_at.desc()).limit(limit)
    )
    detections = result.scalars().all()
    return [
        {
            "id": d.id,
            "plate_text": d.plate_text,
            "confidence": d.confidence,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "camera_id": d.camera_id
        } for d in detections
    ]
    # ==================== 5. سلامت سیستم ====================

@router.get("/health")
async def system_health(
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_current_admin)
):
    """بررسی سلامت سیستم"""
    try:
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }