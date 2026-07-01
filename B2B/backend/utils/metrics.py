"""
metrics.py - جمع‌آوری و ثبت متریک‌های Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from fastapi import Response
import time
from functools import wraps
import psutil
import torch
from typing import Callable

# ==================== متریک‌های پایه ====================

# شمارنده درخواست‌ها
http_requests_total = Counter(
    'http_requests_total',
    'تعداد کل درخواست‌های HTTP',
    ['method', 'endpoint', 'status']
)

# زمان پاسخگویی API
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'زمان پاسخگویی API بر حسب ثانیه',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

# شمارنده تشخیص پلاک
anpr_detections_total = Counter(
    'anpr_detections_total',
    'تعداد کل تشخیص‌های پلاک',
    ['camera', 'confidence_range']
)

# شمارنده خطاهای تشخیص
anpr_detection_errors_total = Counter(
    'anpr_detection_errors_total',
    'تعداد خطاهای تشخیص',
    ['error_type']
)

# میزان اطمینان تشخیص‌ها
anpr_detection_confidence = Histogram(
    'anpr_detection_confidence',
    'توزیع میزان اطمینان تشخیص پلاک',
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
)

# ==================== متریک‌های سیستم ====================

# اطلاعات نسخه سامانه
system_info = Info('anpr_system', 'اطلاعات سامانه ANPR')

# وضعیت دیتابیس
database_status = Gauge('anpr_database_status', 'وضعیت اتصال دیتابیس (1=active, 0=inactive)')

# وضعیت Redis
redis_status = Gauge('anpr_redis_status', 'وضعیت اتصال Redis (1=active, 0=inactive)')

# وضعیت RabbitMQ
rabbitmq_status = Gauge('anpr_rabbitmq_status', 'وضعیت اتصال RabbitMQ (1=active, 0=inactive)')

# ==================== متریک‌های سخت‌افزاری ====================

# مصرف CPU
cpu_usage_percent = Gauge('anpr_cpu_usage_percent', 'مصرف CPU به درصد')

# مصرف حافظه
memory_usage_percent = Gauge('anpr_memory_usage_percent', 'مصرف حافظه به درصد')
memory_usage_bytes = Gauge('anpr_memory_usage_bytes', 'مصرف حافظه به بایت')

# مصرف GPU (در صورت وجود)
gpu_available = Gauge('anpr_gpu_available', 'در دسترس بودن GPU (1=yes, 0=no)')
gpu_utilization_percent = Gauge('anpr_gpu_utilization_percent', 'مصرف GPU به درصد')
gpu_memory_used_mb = Gauge('anpr_gpu_memory_used_mb', 'حافظه GPU استفاده شده به مگابایت')

# ==================== متریک‌های کسب و کار ====================

# تشخیص پلاک به تفکیک سازمان
organization_detections = Counter(
    'anpr_organization_detections_total',
    'تعداد تشخیص پلاک به تفکیک سازمان',
    ['org_id', 'org_name']
)

# پلاک‌های پرتکرار
frequent_plates = Counter(
    'anpr_frequent_plates_total',
    'پلاک‌های پرتکرار',
    ['plate_text']
)

# نرخ موفقیت تشخیص
detection_success_rate = Gauge('anpr_detection_success_rate', 'نرخ موفقیت تشخیص به درصد')


# ==================== دکوریتورها ====================

def track_request(metric_name: str = "http"):
    """دکوریتور برای ردیابی خودکار درخواست‌ها"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, kwargs):
            start_time = time.time()
            method = kwargs.get('request', args[0] if args else None).method if args else "GET"
            endpoint = func.__name__
            
            try:
                result = await func(*args, kwargs)
                status = getattr(result, 'status_code', 200)
                http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
                return result
            except Exception as e:
                http_requests_total.labels(method=method, endpoint=endpoint, status=500).inc()
                raise
            finally:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
        
        return wrapper
    return decorator
def track_detection(camera: str = "unknown"):
    """دکوریتور برای ردیابی تشخیص پلاک"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, kwargs):
            try:
                result = await func(*args, kwargs)
                
                if result and result.get('plates'):
                    for plate in result['plates']:
                        confidence = plate.get('confidence', 0)
                        
                        # تعیین بازه اطمینان
                        if confidence >= 0.95:
                            conf_range = "high"
                        elif confidence >= 0.8:
                            conf_range = "medium"
                        else:
                            conf_range = "low"
                        
                        anpr_detections_total.labels(
                            camera=camera,
                            confidence_range=conf_range
                        ).inc()
                        
                        anpr_detection_confidence.observe(confidence)
                        
                        if plate.get('plate_text'):
                            frequent_plates.labels(plate_text=plate['plate_text']).inc()
                
                return result
            except Exception as e:
                anpr_detection_errors_total.labels(error_type=type(e).__name__).inc()
                raise
        
        return wrapper
    return decorator


# ==================== توابع به‌روزرسانی متریک‌ها ====================

def update_system_metrics():
    """به‌روزرسانی متریک‌های سیستم"""
    # CPU
    cpu_usage_percent.set(psutil.cpu_percent(interval=0.1))
    
    # Memory
    memory = psutil.virtual_memory()
    memory_usage_percent.set(memory.percent)
    memory_usage_bytes.set(memory.used)
    
    # System info
    system_info.info({
        'version': '3.0.0',
        'python_version': '3.10',
        'environment': 'production'
    })


def update_gpu_metrics():
    """به‌روزرسانی متریک‌های GPU"""
    if torch.cuda.is_available():
        gpu_available.set(1)
        gpu_utilization_percent.set(0)  # نیاز به nvidia-ml-py
        gpu_memory_used_mb.set(torch.cuda.memory_allocated() / 1024**2)
    else:
        gpu_available.set(0)


def update_database_metrics(is_connected: bool):
    """به‌روزرسانی متریک دیتابیس"""
    database_status.set(1 if is_connected else 0)


def update_redis_metrics(is_connected: bool):
    """به‌روزرسانی متریک Redis"""
    redis_status.set(1 if is_connected else 0)


def update_rabbitmq_metrics(is_connected: bool):
    """به‌روزرسانی متریک RabbitMQ"""
    rabbitmq_status.set(1 if is_connected else 0)


# ==================== اندپوینت متریک‌ها ====================

async def metrics_endpoint() -> Response:
    """اندپوینت برای Prometheus metrics"""
    update_system_metrics()
    update_gpu_metrics()
    return Response(generate_latest(), media_type="text/plain")