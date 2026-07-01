"""
conftest.py
فایل پیکربندی برای pytest - شامل fixtures و تنظیمات مشترک برای تست‌های پروژه ANPR
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
import json
import numpy as np
import cv2

# افزودن مسیر پروژه به PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== تنظیمات اولیه ====================
@pytest.fixture(scope="session")
def event_loop():
    """ایجاد event loop برای تست‌های async"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== دیتاست تست ====================
@pytest.fixture
def sample_image():
    """ایجاد یک تصویر نمونه برای تست"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    return img


@pytest.fixture
def sample_plate_image():
    """ایجاد یک تصویر نمونه از پلاک (ساده)"""
    img = np.ones((48, 160, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 10), (110, 38), (0, 0, 0), 2)
    return img


@pytest.fixture
def sample_video_frames():
    """ایجاد فریم‌های نمونه برای تست ویدئو"""
    frames = []
    for i in range(30):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 5)
        frames.append(frame)
    return frames


# ==================== دیتاست موقت ====================
@pytest.fixture
def temp_dir():
    """ایجاد پوشه موقت برای تست"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_dataset_dir(temp_dir):
    """ایجاد ساختار دیتاست موقت برای تست"""
    # ایجاد پوشه‌های train/val
    train_img_dir = Path(temp_dir) / "train" / "images"
    train_lbl_dir = Path(temp_dir) / "train" / "labels"
    val_img_dir = Path(temp_dir) / "val" / "images"
    val_lbl_dir = Path(temp_dir) / "val" / "labels"
    
    train_img_dir.mkdir(parents=True)
    train_lbl_dir.mkdir(parents=True)
    val_img_dir.mkdir(parents=True)
    val_lbl_dir.mkdir(parents=True)
    
    # ایجاد یک تصویر نمونه
    sample_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(train_img_dir / "test_001.jpg"), sample_img)
    cv2.imwrite(str(val_img_dir / "test_001.jpg"), sample_img)
    
    # ایجاد فایل برچسب YOLO
    with open(train_lbl_dir / "test_001.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.1")
    
    with open(val_lbl_dir / "test_001.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.1")
    
    # ایجاد فایل data.yaml
    data_yaml = {
        "path": str(temp_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["license_plate"]
    }
    import yaml
    with open(Path(temp_dir) / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    
    return temp_dir


# ==================== ماژول‌های تست ====================
@pytest.fixture
def mock_detector():
    """Mock مدل تشخیص برای تست"""
    class MockDetector:
        def detect(self, image):
            return [{"bbox": [10, 10, 50, 30], "confidence": 0.95, "class": 0}]
        
        def draw_annotations(self, image, plates):
            return image
    
    return MockDetector()


@pytest.fixture
def mock_ocr():
    """Mock مدل OCR برای تست"""
    class MockOCR:
        def read(self, image):
            return "۱۲۳۴۵۶۷"
    
    return MockOCR()


# ==================== تست API ====================
@pytest.fixture
def test_client():
    """ایجاد client برای تست FastAPI"""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """ایجاد هدر احراز هویت برای تست"""
    return {"Authorization": "Bearer test_token_123"}


# ==================== دیتای نمونه ====================
@pytest.fixture
def sample_plate_data():
    """داده نمونه پلاک برای تست"""
    return {
        "plate_text": "۱۲۳۴۵۶۷",
        "confidence": 0.95,
        "bbox": [100, 200, 300, 400],
        "timestamp": "2024-01-01T12:00:00"
    }
    """
conftest.py
فایل پیکربندی برای pytest - شامل fixtures و تنظیمات مشترک برای تست‌های پروژه ANPR
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
import json
import numpy as np
import cv2

# افزودن مسیر پروژه به PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== تنظیمات اولیه ====================
@pytest.fixture(scope="session")
def event_loop():
    """ایجاد event loop برای تست‌های async"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== دیتاست تست ====================
@pytest.fixture
def sample_image():
    """ایجاد یک تصویر نمونه برای تست"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    return img


@pytest.fixture
def sample_plate_image():
    """ایجاد یک تصویر نمونه از پلاک (ساده)"""
    img = np.ones((48, 160, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 10), (110, 38), (0, 0, 0), 2)
    return img


@pytest.fixture
def sample_video_frames():
    """ایجاد فریم‌های نمونه برای تست ویدئو"""
    frames = []
    for i in range(30):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 5)
        frames.append(frame)
    return frames


# ==================== دیتاست موقت ====================
@pytest.fixture
def temp_dir():
    """ایجاد پوشه موقت برای تست"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_dataset_dir(temp_dir):
    """ایجاد ساختار دیتاست موقت برای تست"""
    # ایجاد پوشه‌های train/val
    train_img_dir = Path(temp_dir) / "train" / "images"
    train_lbl_dir = Path(temp_dir) / "train" / "labels"
    val_img_dir = Path(temp_dir) / "val" / "images"
    val_lbl_dir = Path(temp_dir) / "val" / "labels"
    
    train_img_dir.mkdir(parents=True)
    train_lbl_dir.mkdir(parents=True)
    val_img_dir.mkdir(parents=True)
    val_lbl_dir.mkdir(parents=True)
    
    # ایجاد یک تصویر نمونه
    sample_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(train_img_dir / "test_001.jpg"), sample_img)
    cv2.imwrite(str(val_img_dir / "test_001.jpg"), sample_img)
    
    # ایجاد فایل برچسب YOLO
    with open(train_lbl_dir / "test_001.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.1")
    
    with open(val_lbl_dir / "test_001.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.1")
    
    # ایجاد فایل data.yaml
    data_yaml = {
        "path": str(temp_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["license_plate"]
    }
    import yaml
    with open(Path(temp_dir) / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    
    return temp_dir


# ==================== ماژول‌های تست ====================
@pytest.fixture
def mock_detector():
    """Mock مدل تشخیص برای تست"""
    class MockDetector:
        def detect(self, image):
            return [{"bbox": [10, 10, 50, 30], "confidence": 0.95, "class": 0}]
        
        def draw_annotations(self, image, plates):
            return image
    
    return MockDetector()


@pytest.fixture
def mock_ocr():
    """Mock مدل OCR برای تست"""
    class MockOCR:
        def read(self, image):
            return "۱۲۳۴۵۶۷"
    
    return MockOCR()


# ==================== تست API ====================
@pytest.fixture
def test_client():
    """ایجاد client برای تست FastAPI"""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """ایجاد هدر احراز هویت برای تست"""
    return {"Authorization": "Bearer test_token_123"}


# ==================== دیتای نمونه ====================
@pytest.fixture
def sample_plate_data():
    """داده نمونه پلاک برای تست"""
    return {
        "plate_text": "۱۲۳۴۵۶۷",
        "confidence": 0.95,
        "bbox": [100, 200, 300, 400],
        "timestamp": "2024-01-01T12:00:00"
    }
    @pytest.fixture
    def sample_organization():
        """داده نمونه سازمان برای تست"""
    return {
        "id": 1,
        "name": "سازمان تست",
        "tier": "pro",
        "max_cameras": 10,
        "quota_limit": 50000,
        "is_active": True
    }


@pytest.fixture
def sample_user():
    """داده نمونه کاربر برای تست"""
    return {
        "id": 1,
        "org_id": 1,
        "email": "test@anpr.ir",
        "full_name": "کاربر تست",
        "role": "admin",
        "is_active": True
    }


@pytest.fixture
def sample_camera():
    """داده نمونه دوربین برای تست"""
    return {
        "id": 1,
        "org_id": 1,
        "name": "دوربین تست",
        "stream_url": "rtsp://test.stream",
        "stream_type": "rtsp",
        "is_active": True
    }


# ==================== اتصال به دیتابیس تست ====================
@pytest.fixture
async def test_db():
    """ایجاد اتصال به دیتابیس تست (با SQLite in-memory)"""
    import asyncpg
    import asyncio
    
    # برای تست از دیتابیس in-memory یا یک دیتابیس موقت استفاده کنید
    # در اینجا یک mock ساده برگردانده می‌شود
    class MockDB:
        async def fetch(self, query, *args):
            return []
        
        async def fetchrow(self, query, *args):
            return None
        
        async def execute(self, query, *args):
            return "OK"
    
    return MockDB()


# ==================== کش‌های تست ====================
@pytest.fixture
def mock_redis():
    """Mock Redis برای تست"""
    class MockRedis:
        def init(self):
            self.data = {}
        
        async def get(self, key):
            return self.data.get(key)
        
        async def set(self, key, value):
            self.data[key] = value
        
        async def delete(self, key):
            if key in self.data:
                del self.data[key]
        
        async def exists(self, key):
            return key in self.data
    
    return MockRedis()


# ==================== پیکربندی pytest ====================
def pytest_configure(config):
    """تنظیمات قبل از اجرای تست‌ها"""
    os.environ["TESTING"] = "True"
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"


def pytest_unconfigure(config):
    """تنظیمات بعد از اجرای تست‌ها"""
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


# ==================== مارکرهای تست ====================
def pytest_addoption(parser):
    """اضافه کردن گزینه‌های خط فرمان به pytest"""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="اجرای تست‌های کند"
    )
    parser.addoption(
        "--run-gpu", action="store_true", default=False, help="اجرای تست‌های نیازمند GPU"
    )


def pytest_collection_modifyitems(config, items):
    """اصلاح تست‌ها بر اساس مارکرها"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="نیاز به --run-slow دارد")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="نیاز به --run-gpu دارد")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ==================== Mock داده ====================
@pytest.fixture
def mock_detection_result():
    """نتیجه mock تشخیص پلاک"""
    return {
        "success": True,
        "plates": [
            {"plate_text": "۱۲۳۴۵۶۷", "confidence": 0.95, "bbox": [100, 200, 300, 400]}
        ],
        "num_plates": 1,
        "processing_time_ms": 45.2
    }


@pytest.fixture
def mock_video_processing_result():
    """نتیجه mock پردازش ویدئو"""
    return {
        "success": True,
        "total_frames": 300,
        "detections": [
            {"frame": 10, "plate_text": "۱۲۳۴۵۶۷", "confidence": 0.95},
            {"frame": 45, "plate_text": "۸۹۱۲۳۴۵", "confidence": 0.92}
        ],
        "num_detections": 2,
        "processing_time_seconds": 12.5
    }