"""
model_version.py - مدیریت نسخه‌های مدل‌ها
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """مدیریت نسخه‌های مدل‌های هوش مصنوعی"""
    
    def init(self, models_dir: str = "models", metadata_file: str = "models/model_metadata.json"):
        self.models_dir = Path(models_dir)
        self.metadata_file = Path(metadata_file)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """بارگذاری متادیتا"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._create_default_metadata()
    
    def _create_default_metadata(self) -> Dict:
        """ایجاد متادیتای پیش‌فرض"""
        return {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "models": {},
            "deployment_history": [],
            "active_models": {}
        }
    
    def _save_metadata(self):
        """ذخیره متادیتا"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def get_active_version(self, model_name: str) -> str:
        """دریافت نسخه فعال مدل"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"مدل {model_name} یافت نشد")
        return self.metadata["models"][model_name]["active_version"]
    
    def get_model_file(self, model_name: str, version: Optional[str] = None) -> Path:
        """دریافت مسیر فایل مدل"""
        if version is None:
            version = self.get_active_version(model_name)
        
        if model_name not in self.metadata["models"]:
            raise ValueError(f"مدل {model_name} یافت نشد")
        
        if version not in self.metadata["models"][model_name]["versions"]:
            raise ValueError(f"نسخه {version} برای مدل {model_name} یافت نشد")
        
        file_name = self.metadata["models"][model_name]["versions"][version]["file"]
        return self.models_dir / file_name
    
    def switch_version(self, model_name: str, new_version: str, deployed_by: str = "system"):
        """تغییر نسخه فعال مدل"""
        if model_name not in self.metadata["models"]:
            raise ValueError(f"مدل {model_name} یافت نشد")
        
        old_version = self.metadata["models"][model_name].get("active_version")
        
        if new_version not in self.metadata["models"][model_name]["versions"]:
            raise ValueError(f"نسخه {new_version} برای مدل {model_name} یافت نشد")
        
        self.metadata["models"][model_name]["active_version"] = new_version
        
        self.metadata["deployment_history"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "from_version": old_version,
            "to_version": new_version,
            "deployed_by": deployed_by
        })
        
        self._save_metadata()
        logger.info(f"مدل {model_name} از نسخه {old_version} به {new_version} تغییر کرد")
    
    def register_version(self, model_name: str, version: str, file_path: Path, metrics: Dict):
        """ثبت نسخه جدید مدل"""
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {
                "active_version": version,
                "versions": {}
            }
        
        self.metadata["models"][model_name]["versions"][version] = {
            "file": file_path.name,
            "size_mb": file_path.stat().st_size / (1024 * 1024),
            "registered_at": datetime.now().isoformat(),
            "status": "registered",
            **metrics
        }
        
        self._save_metadata()
        logger.info(f"نسخه {version} برای مدل {model_name} ثبت شد")
    
    def get_model_accuracy(self, model_name: str, version: Optional[str] = None) -> float:
        """دریافت دقت مدل"""
        if version is None:
            version = self.get_active_version(model_name)
        
        return self.metadata["models"][model_name]["versions"][version].get("accuracy", 0.0)