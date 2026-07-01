"""
deep_core.py - هسته یادگیری عمیق و مدیریت مدل‌ها
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class DeepLearningCore:
    """کلاس اصلی مدیریت مدل‌های یادگیری عمیق"""
    
    def init(self, device: Optional[str] = None):
        """
        Args:
            device: دستگاهی که مدل روی آن اجرا می‌شود ('cuda', 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"دستگاه محاسباتی: {self.device}")
        
        self.models: Dict[str, nn.Module] = {}
    
    def load_yolo(self, model_path: str, task: str = 'detect') -> YOLO:
        """
        بارگذاری مدل YOLO
        
        Args:
            model_path: مسیر فایل مدل
            task: نوع وظیفه ('detect', 'segment', 'classify')
        
        Returns:
            مدل YOLO
        """
        try:
            model = YOLO(model_path, task=task)
            model.to(self.device)
            logger.info(f"مدل YOLO از {model_path} بارگذاری شد")
            return model
        except Exception as e:
            logger.error(f"خطا در بارگذاری YOLO: {e}")
            raise
    
    def train_yolo(
        self,
        data_yaml: str,
        model_name: str = 'yolov8n.pt',
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        lr: float = 0.01,
        device: str = 'cuda',
        project: str = 'models',
        name: str = 'plate_detector'
    ) -> Dict[str, Any]:
        """
        آموزش مدل YOLO
        
        Returns:
            نتایج آموزش
        """
        model = YOLO(model_name)
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr,
            device=device,
            project=project,
            name=name,
            exist_ok=True
        )
        
        logger.info(f"آموزش YOLO به پایان رسید. نتایج در {project}/{name}")
        return results
    
    def load_crnn(self, model_path: str, num_classes: int = 35, input_height: int = 48, input_width: int = 160) -> nn.Module:
        """
        بارگذاری مدل CRNN (برای OCR فارسی)
        
        Args:
            model_path: مسیر فایل مدل
            num_classes: تعداد کلاس‌ها (حروف + اعداد + blank)
            input_height: ارتفاع ورودی
            input_width: عرض ورودی
        
        Returns:
            مدل CRNN
        """
        class CRNN(nn.Module):
            def init(self, num_classes, hidden_size=256):
                super().init()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                )
                self.rnn = nn.LSTM(256 * (input_height // 8), hidden_size, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(hidden_size * 2, num_classes)
            
            def forward(self, x):
                features = self.cnn(x)
                features = features.permute(0, 3, 1, 2)
                features = features.reshape(features.size(0), features.size(1), -1)
                out, _ = self.rnn(features)
                return self.fc(out)
        
        model = CRNN(num_classes)
        
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"مدل CRNN از {model_path} بارگذاری شد")
        else:
            logger.warning(f"فایل {model_path} یافت نشد، مدل با وزن‌های تصادفی ایجاد شد")
            model.to(self.device)
        model.eval()
        
        return model
    
    def load_lstm(self, model_path: str, input_size: int = 4, hidden_size: int = 128, num_layers: int = 2) -> nn.Module:
        """
        بارگذاری مدل LSTM برای تشخیص ناهنجاری
        
        Args:
            model_path: مسیر فایل مدل
            input_size: تعداد ویژگی‌های ورودی
            hidden_size: اندازه لایه پنهان
            num_layers: تعداد لایه‌های LSTM
        
        Returns:
            مدل LSTM
        """
        class LSTMAnomaly(nn.Module):
            def init(self, input_size, hidden_size, num_layers):
                super().init()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.fc(out)
                return self.sigmoid(out)
        
        model = LSTMAnomaly(input_size, hidden_size, num_layers)
        
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"مدل LSTM از {model_path} بارگذاری شد")
        else:
            logger.warning(f"فایل {model_path} یافت نشد، مدل با وزن‌های تصادفی ایجاد شد")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_device_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات دستگاه محاسباتی"""
        info = {
            "device": self.device,
            "torch_version": torch.version,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        
        return info