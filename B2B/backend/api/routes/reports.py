"""
reports.py - مسیرهای گزارش‌گیری و آمار
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Optional
from datetime import datetime, date, timedelta

from core.database import get_db
from core.security import get_current_user
from models import Detection

router = APIRouter(prefix="/reports", tags=["گزارش‌ها"])


@router.get("/detections")
async def get_detections_report(
    start_date: date = Query(..., description="تاریخ شروع"),
    end_date: date = Query(..., description="تاریخ پایان"),
    org_id: Optional[int] = None,
    camera_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """دریافت گزارش تشخیص‌ها در بازه زمانی"""
    # اگر کاربر ادمین نباشد، فقط سازمان خودش
    if current_user["role"] not in ["super_admin", "admin"]:
        # باید org_id کاربر را از دیتابیس بگیریم
        # در اینجا فرض می‌کنیم org_id در توکن ذخیره شده یا از دیتابیس خوانده شود
        pass
    
    query = select(Detection).where(
        and_(
            Detection.created_at >= start_date,
            Detection.created_at < end_date + timedelta(days=1)
        )
    )
    if org_id:
        query = query.where(Detection.org_id == org_id)
    if camera_id:
        query = query.where(Detection.camera_id == camera_id)
    
    query = query.order_by(Detection.created_at.desc())
    result = await db.execute(query)
    detections = result.scalars().all()
    
    return [
        {
            "id": d.id,
            "plate_text": d.plate_text,
            "confidence": d.confidence,
            "created_at": d.created_at,
            "camera_id": d.camera_id,
            "track_id": d.track_id,
            "anomaly_score": d.anomaly_score
        } for d in detections
    ]


@router.get("/statistics")
async def get_statistics(
    start_date: date,
    end_date: date,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """آمار خلاصه تشخیص‌ها"""
    query = select(
        func.count(Detection.id).label("total"),
        func.avg(Detection.confidence).label("avg_conf"),
        func.count(func.distinct(Detection.plate_text)).label("unique_plates")
    ).where(
        and_(
            Detection.created_at >= start_date,
            Detection.created_at < end_date + timedelta(days=1)
        )
    )
    result = await db.execute(query)
    row = result.one()
    return {
        "total_detections": row.total or 0,
        "average_confidence": float(row.avg_conf) if row.avg_conf else 0,
        "unique_plates": row.unique_plates or 0,
        "start_date": start_date,
        "end_date": end_date
    }