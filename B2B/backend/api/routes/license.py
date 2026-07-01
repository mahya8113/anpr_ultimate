"""
license.py - مسیرهای مدیریت لایسنس
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

from core.licensing import LicenseManager
from core.security import get_current_admin

router = APIRouter(prefix="/license", tags=["لایسنس"])


class LicenseGenerateRequest(BaseModel):
    org_id: int
    max_cameras: int
    expiry_days: int


class LicenseValidateResponse(BaseModel):
    valid: bool
    org_id: Optional[int] = None
    max_cameras: Optional[int] = None
    expires_at: Optional[datetime] = None
    days_remaining: Optional[int] = None
    error: Optional[str] = None


@router.post("/generate")
async def generate_license(
    req: LicenseGenerateRequest,
    admin=Depends(get_current_admin)
):
    """تولید لایسنس جدید (فقط ادمین)"""
    token = LicenseManager.generate_license(
        org_id=req.org_id,
        max_cameras=req.max_cameras,
        expiry_days=req.expiry_days
    )
    return {"license_token": token}


@router.post("/validate", response_model=LicenseValidateResponse)
async def validate_license(
    token: str,
    org_id: Optional[int] = None
):
    """اعتبارسنجی لایسنس (بدون نیاز به احراز هویت)"""
    try:
        payload = LicenseManager.validate_license(token)
        if org_id and payload["org_id"] != org_id:
            return LicenseValidateResponse(
                valid=False,
                error="لایسنس برای این سازمان معتبر نیست"
            )
        exp = datetime.fromtimestamp(payload["exp"])
        remaining = max(0, (exp - datetime.now()).days)
        return LicenseValidateResponse(
            valid=True,
            org_id=payload["org_id"],
            max_cameras=payload["max_cameras"],
            expires_at=exp,
            days_remaining=remaining
        )
    except Exception as e:
        return LicenseValidateResponse(valid=False, error=str(e))