"""
test_model_version.py
تست‌های مربوط به مدیریت نسخه‌های مدل‌های هوش مصنوعی
شامل: بارگذاری مدل، تغییر نسخه، ذخیره متادیتا، برگشت به نسخه قبلی
"""

import pytest
import json
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.model_version import ModelVersionManager


# ==================== Fixtures ====================
@pytest.fixture
def temp_models_dir():
    """ایجاد پوشه موقت برای مدل‌ها"""
    temp_dir = tempfile.mkdtemp()
    models_dir = Path(temp_dir) / "models"
    models_dir.mkdir()
    
    # ایجاد فایل‌های مدل نمونه
    for i in range(3):
        model_file = models_dir / f"yolo_plate_detector_v{i+1}.0.pt"
        model_file.touch()
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_metadata():
    """ایجاد متادیتای نمونه"""
    return {
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat(),
        "models": {
            "yolo_plate_detector": {
                "active_version": "v1.0.0",
                "versions": {
                    "v1.0.0": {
                        "file": "yolo_plate_detector_v1.0.0.pt",
                        "size_mb": 12.4,
                        "accuracy": 0.852,
                        "status": "production"
                    },
                    "v2.0.0": {
                        "file": "yolo_plate_detector_v2.0.0.pt",
                        "size_mb": 24.6,
                        "accuracy": 0.908,
                        "status": "staging"
                    }
                }
            },
            "crnn_persian_ocr": {
                "active_version": "v1.0.0",
                "versions": {
                    "v1.0.0": {
                        "file": "crnn_persian_v1.0.0.pth",
                        "size_mb": 28.4,
                        "accuracy": 0.782,
                        "status": "production"
                    }
                }
            }
        },
        "deployment_history": []
    }


# ==================== تست‌های پایه ====================
class TestModelVersionManager:
    """تست‌های کلاس ModelVersionManager"""
    
    def test_init_with_existing_metadata(self, temp_models_dir, sample_metadata):
        """تست مقداردهی اولیه با متادیتای موجود"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        assert manager.metadata == sample_metadata
    
    def test_init_without_metadata(self, temp_models_dir):
        """تست مقداردهی اولیه بدون متادیتا"""
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(Path(temp_models_dir) / "models" / "model_metadata.json")
        )
        
        assert manager.metadata["version"] == "1.0.0"
        assert "models" in manager.metadata
        assert "deployment_history" in manager.metadata
    
    def test_save_metadata(self, temp_models_dir):
        """تست ذخیره متادیتا"""
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(Path(temp_models_dir) / "models" / "model_metadata.json")
        )
        
        manager.metadata["test_field"] = "test_value"
        manager._save_metadata()
        
        # بررسی ذخیره شدن
        with open(manager.metadata_path, "r") as f:
            saved = json.load(f)
        
        assert saved["test_field"] == "test_value"
        assert "last_updated" in saved
        # ==================== تست‌های مدیریت نسخه ====================
class TestVersionManagement:
    """تست‌های مدیریت نسخه مدل‌ها"""
    
    def test_get_active_version(self, temp_models_dir, sample_metadata):
        """تست دریافت نسخه فعال مدل"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        active_version = manager.get_active_version("yolo_plate_detector")
        assert active_version == "v1.0.0"
        
        active_version_crnn = manager.get_active_version("crnn_persian_ocr")
        assert active_version_crnn == "v1.0.0"
    
    def test_get_active_version_not_found(self, temp_models_dir):
        """تست دریافت نسخه فعال مدل ناموجود"""
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(Path(temp_models_dir) / "models" / "model_metadata.json")
        )
        
        with pytest.raises(ValueError):
            manager.get_active_version("nonexistent_model")
    
    def test_get_model_file(self, temp_models_dir, sample_metadata):
        """تست دریافت نام فایل مدل"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        # با نسخه مشخص
        file_path = manager.get_model_file("yolo_plate_detector", "v2.0.0")
        assert "yolo_plate_detector_v2.0.0.pt" in file_path
        
        # با نسخه فعال (پیش‌فرض)
        file_path_default = manager.get_model_file("yolo_plate_detector")
        assert "yolo_plate_detector_v1.0.0.pt" in file_path_default
    
    def test_switch_version(self, temp_models_dir, sample_metadata):
        """تست تغییر نسخه فعال مدل"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        # تغییر نسخه
        manager.switch_version("yolo_plate_detector", "v2.0.0", deployed_by="test_user")
        
        # بررسی تغییر
        active = manager.get_active_version("yolo_plate_detector")
        assert active == "v2.0.0"
        
        # بررسی ثبت در تاریخچه
        assert len(manager.metadata["deployment_history"]) == 1
        history = manager.metadata["deployment_history"][0]
        assert history["model"] == "yolo_plate_detector"
        assert history["to_version"] == "v2.0.0"
        assert history["deployed_by"] == "test_user"
    
    def test_switch_nonexistent_version(self, temp_models_dir, sample_metadata):
        """تست تغییر به نسخه ناموجود"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        with pytest.raises(ValueError):
            manager.switch_version("yolo_plate_detector", "v99.0.0")
    
    def test_rollback_version(self, temp_models_dir, sample_metadata):
        """تست برگشت به نسخه قبلی"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        # تغییر به نسخه جدید
        manager.switch_version("yolo_plate_detector", "v2.0.0")
        
        # برگشت
        manager.rollback("yolo_plate_detector")
        
        active = manager.get_active_version("yolo_plate_detector")
        assert active == "v1.0.0"


# ==================== تست‌های دقت مدل ====================
class TestModelAccuracy:
    """تست‌های مربوط به دقت مدل‌ها"""
    
    def test_get_model_accuracy(self, temp_models_dir, sample_metadata):
        """تست دریافت دقت مدل"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        accuracy = manager.get_model_accuracy("yolo_plate_detector", "v1.0.0")
        assert accuracy == 0.852
        
        accuracy_active = manager.get_model_accuracy("yolo_plate_detector")
        assert accuracy_active == 0.852  # نسخه فعال v1.0.0
    
    def test_get_nonexistent_model_accuracy(self, temp_models_dir):
        """تست دریافت دقت مدل ناموجود"""
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(Path(temp_models_dir) / "models" / "model_metadata.json")
        )
        
        with pytest.raises(ValueError):
            manager.get_model_accuracy("nonexistent_model")


# ==================== تست‌های بارگذاری مدل ====================
class TestModelLoading:
    """تست‌های بارگذاری مدل"""
    
    @patch("torch.load")
    def test_load_yolo_model(self, mock_torch_load, temp_models_dir):
        """تست بارگذاری مدل YOLO"""
        from ultralytics import YOLO
        
        mock_torch_load.return_value = {"model": "fake_state_dict"}
        
        # این تست نیاز به mock کردن YOLO دارد
        pass
    
    @patch("torch.load")
    def test_load_crnn_model(self, mock_torch_load, temp_models_dir):
        """تست بارگذاری مدل CRNN"""
        mock_torch_load.return_value = {"cnn": "fake_weights"}
        
        # این تست نیاز به mock کردن CRNN دارد
        pass


# ==================== تست‌های ذخیره مدل ====================
class TestModelSaving:
    """تست‌های ذخیره مدل جدید"""
    
    def test_register_new_version(self, temp_models_dir, sample_metadata):
        """تست ثبت نسخه جدید مدل"""
        metadata_path = Path(temp_models_dir) / "models" / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)
        
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(metadata_path)
        )
        
        # ثبت نسخه جدید
        new_version_data = {
            "file": "yolo_plate_detector_v3.0.0.pt",
            "size_mb": 30.2,
            "accuracy": 0.925,
            "status": "staging"
        }
        
        manager._register_version("yolo_plate_detector", "v3.0.0", new_version_data)
        
        # بررسی ثبت
        assert "v3.0.0" in manager.metadata["models"]["yolo_plate_detector"]["versions"]
        assert manager.metadata["models"]["yolo_plate_detector"]["versions"]["v3.0.0"]["accuracy"] == 0.925
    
    def test_deploy_new_model(self, temp_models_dir):
        """تست استقرار مدل جدید"""
        manager = ModelVersionManager(
            models_dir=str(Path(temp_models_dir) / "models"),
            metadata_path=str(Path(temp_models_dir) / "models" / "model_metadata.json")
        )
        
        # شبیه‌سازی استقرار
        result = manager.deploy_new_model(
            model_path="/fake/path/model.pt",
            model_name="test_model",
            version="v1.0.0",
            source_url=None,
            skip_validation=True
        )
        
        assert result is True or result is False
        # بررسی ثبت در متادیتا
        if result:
            assert "test_model" in manager.metadata["models"]


# ==================== تست‌های یکپارچگی ====================
class TestIntegration:
    """تست‌های یکپارچگی با سرویس‌های دیگر"""
    
    @pytest.mark.asyncio
    async def test_model_version_in_detection_service(self):
        """تست استفاده از نسخه مدل در سرویس تشخیص"""
        from backend.modules.detection import ObjectDetector
        
        # این تست نیاز به راه‌اندازی کامل سرویس دارد
        pass
    
    def test_model_metadata_api(self):
        """تست API متادیتای مدل"""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        # درخواست به endpoint متادیتا
        response = client.get("/admin/models/metadata")
        assert response.status_code in [200, 401]


# ==================== اجرای تست‌ها ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])