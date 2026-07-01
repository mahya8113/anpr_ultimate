"""
main.py - نقطه ورود اصلی سرویس بک‌اند FastAPI
سامانه تشخیص پلاک فارسی (ANPR)
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import os
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به PATH
sys.path.insert(0, str(Path(__file__).parent))

# ==================== تنظیمات لاگ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger("anpr-backend")

# ==================== تنظیمات ====================
from core.config import settings
from core.database import init_db, close_db
from core.redis_client import redis_client
from core.rabbitmq_client import rabbitmq_client

# ==================== رویدادهای چرخه حیات ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    مدیریت چرخه حیات اپلیکیشن
    - startup: اتصال به دیتابیس، Redis، RabbitMQ، بارگذاری مدل‌ها
    - shutdown: بستن اتصالات
    """
    # ========== Startup ==========
    logger.info("🚀 در حال راه‌اندازی سامانه ANPR...")
    
    # ایجاد پوشه‌های مورد نیاز
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # اتصال به دیتابیس
    try:
        await init_db()
        logger.info("✅ اتصال به PostgreSQL برقرار شد")
    except Exception as e:
        logger.error(f"❌ خطا در اتصال به PostgreSQL: {e}")
    
    # اتصال به Redis
    try:
        await redis_client.ping()
        logger.info("✅ اتصال به Redis برقرار شد")
    except Exception as e:
        logger.error(f"❌ خطا در اتصال به Redis: {e}")
    
    # اتصال به RabbitMQ
    try:
        await rabbitmq_client.connect()
        logger.info("✅ اتصال به RabbitMQ برقرار شد")
    except Exception as e:
        logger.error(f"❌ خطا در اتصال به RabbitMQ: {e}")
    
    # بارگذاری مدل‌ها
    try:
        from modules.detection import ObjectDetector
        from modules.advanced_ocr import AdvancedOCR
        
        app.state.detector = ObjectDetector()
        app.state.ocr = AdvancedOCR()
        logger.info("✅ مدل‌های هوش مصنوعی بارگذاری شدند")
    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری مدل‌ها: {e}")
    
    logger.info("🎉 سامانه ANPR با موفقیت راه‌اندازی شد!")
    
    yield
    
    # ========== Shutdown ==========
    logger.info("🛑 در حال توقف سامانه ANPR...")
    
    # بستن اتصال دیتابیس
    await close_db()
    logger.info("✅ اتصال PostgreSQL بسته شد")
    
    # بستن اتصال Redis
    await redis_client.close()
    logger.info("✅ اتصال Redis بسته شد")
    
    # بستن اتصال RabbitMQ
    await rabbitmq_client.close()
    logger.info("✅ اتصال RabbitMQ بسته شد")
    
    logger.info("👋 سامانه ANPR متوقف شد!")


# ==================== ایجاد اپلیکیشن FastAPI ====================
app = FastAPI(
    title="ANPR Ultimate API",
    description="سامانه هوشمند تشخیص پلاک خودروهای ایران",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ==================== Middlewareها ====================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== هندلرهای خطا ====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """هندلر خطاهای اعتبارسنجی"""
    logger.error(f"خطای اعتبارسنجی: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "خطا در اعتبارسنجی داده‌ها",
            "errors": exc.errors(),
            "message": "لطفاً مقادیر ورودی را بررسی کنید"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """هندلر خطاهای HTTP"""
    logger.warning(f"خطای HTTP: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "message": "خطا در پردازش درخواست"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """هندلر خطاهای عمومی"""
    logger.error(f"خطای داخلی سرور: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "خطای داخلی سرور",
            "message": "لطفاً با پشتیبانی تماس بگیرید"
        }
    )


# ==================== Middleware سفارشی ====================
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """افزودن هدر زمان پردازش به پاسخ"""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """لاگ کردن درخواست‌ها"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


# ==================== روت‌های عمومی ====================
@app.get("/", tags=["General"])
async def root():
    """صفحه اصلی API"""
    return {
        "message": "🚗 سامانه هوشمند تشخیص پلاک خودروهای ایران",
        "version": "3.0.0",
        "status": "running",
        "docs_url": "/docs",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["General"])
async def health_check():
    """بررسی سلامت سامانه"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "unknown",
            "redis": "unknown",
            "rabbitmq": "unknown"
        }
    }
    
    # بررسی دیتابیس
    try:
        from core.database import get_db_connection
        conn = await get_db_connection()
        await conn.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # بررسی Redis
    try:
        await redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # بررسی RabbitMQ
    try:
        if rabbitmq_client.connection and rabbitmq_client.connection.is_open:
            health_status["services"]["rabbitmq"] = "healthy"
        else:
            health_status["services"]["rabbitmq"] = "disconnected"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["rabbitmq"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/info", tags=["General"])
async def system_info():
    """اطلاعات سیستم"""
    import psutil
    import torch
    
    return {
        "system": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "hardware": {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        } if torch.cuda.is_available() else {"available": False},
        "timestamp": datetime.now().isoformat()
    }


# ==================== import روت‌ها ====================
from api.routes import auth, detect, video, reports, admin, license as license_routes  # noqa: E402
from api.websocket import live  # noqa: E402

# ثبت روت‌ها
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(detect.router, prefix="/detect", tags=["Detection"])
app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(license_routes.router, prefix="/license", tags=["License"])
app.include_router(live.router, prefix="/ws", tags=["WebSocket"])


# ==================== اجرای مستقیم ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
        workers=4 if not settings.DEBUG else 1
    )