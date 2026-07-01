"""
lowlight_ir.py - بهبود تصاویر کمنور و فیوژن با تصاویر مادون قرمز (IR)
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LowLightEnhancer:
    @staticmethod
    def zero_dce_enhance(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        gamma = 1.8
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(enhanced, table)

    @staticmethod
    def fuse_ir_rgb(ir_img: np.ndarray, rgb_img: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        ir_resized = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))
        ir_norm = cv2.normalize(ir_resized, None, 0, 1, cv2.NORM_MINMAX)
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        fused = cv2.addWeighted(rgb_norm, alpha, ir_norm, 1 - alpha, 0)
        return (fused * 255).astype(np.uint8)

    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image)

    @staticmethod
    def retinex_enhance(image: np.ndarray, sigma: float = 30) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        log_img = np.log(img_float + 0.01)
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log_blur = np.log(blur + 0.01)
        retinex = log_img - log_blur
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min())
        return (retinex * 255).astype(np.uint8)

    @staticmethod
    def enhance_video(video_path: str, output_path: str, method: str = 'clahe'):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(3)), int(cap.get(4))))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if method == 'clahe':
                enhanced = LowLightEnhancer.zero_dce_enhance(frame)
            elif method == 'gamma':
                enhanced = LowLightEnhancer.gamma_correction(frame)
            elif method == 'hist':
                enhanced = LowLightEnhancer.histogram_equalization(frame)
            else:
                enhanced = frame
            out.write(enhanced)
        cap.release()
        out.release()
        logger.info(f"Enhanced video saved to {output_path}")