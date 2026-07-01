"""
depth_estimation.py - تخمین عمق صحنه با استفاده از مدل MiDaS
"""

import torch
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    def init(self, model_type: str = "DPT_Hybrid", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None
        self.model_type = model_type
        self._load_model()

    def _load_model(self):
        try:
            import torchvision.transforms as transforms
            # MiDaS
            midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model = midas.to(self.device)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info(f"Depth estimation model {self.model_type} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.model = None

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        # image: BGR numpy
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.model(img_tensor)
        depth = depth.squeeze().cpu().numpy()
        # نرمال‌سازی
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)
        return depth

    def get_3d_points(self, depth_map: np.ndarray, fx: float = 700, fy: float = 700, cx: float = None, cy: float = None) -> np.ndarray:
        h, w = depth_map.shape
        if cx is None:
            cx = w / 2
        if cy is None:
            cy = h / 2
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map.astype(np.float32) / 255.0 * 10.0  # scale to 10 meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=-1)