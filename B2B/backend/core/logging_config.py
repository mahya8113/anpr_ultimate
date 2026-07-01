"""
logging_config.py - پیکربندی لاگ سیستم
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .config import settings

# ایجاد پوشه لاگ
Path("logs").mkdir(exist_ok=True)


def setup_logging():
    """پیکربندی سیستم لاگ"""
    
    # فرمت لاگ
    if settings.LOG_FORMAT == "json":
        from pythonjsonlogger import jsonlogger
        
        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super().add_fields(log_record, record, message_dict)
                log_record['level'] = record.levelname
                log_record['name'] = record.name
    
        formatter = CustomJsonFormatter('%(asctime)s %(level)s %(name)s %(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # هندلر کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    # هندلر فایل (چرخشی)
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10_485_760,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # هندلر خطاها (فایل جداگانه)
    error_handler = RotatingFileHandler(
        "logs/error.log",
        maxBytes=10_485_760,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    
    # تنظیم root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # کاهش لاگ کتابخانه‌های پرحرف
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aio_pika").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    
    logging.info("🚀 سیستم لاگ راه‌اندازی شد")


def get_logger(name: str) -> logging.Logger:
    """دریافت logger با نام مشخص"""
    return logging.getLogger(name)


class RequestLogger:
    """لاگ درخواست‌های HTTP"""
    
    def init(self):
        self.logger = get_logger("http")
    
    def log_request(self, method: str, path: str, status: int, duration: float, client_ip: str):
        self.logger.info(
            f"Request: {method} {path} | Status: {status} | Duration: {duration:.3f}s | IP: {client_ip}"
        )
    
    def log_error(self, method: str, path: str, error: str, client_ip: str):
        self.logger.error(f"Error: {method} {path} | Error: {error} | IP: {client_ip}")


request_logger = RequestLogger()