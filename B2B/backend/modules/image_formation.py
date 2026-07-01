"""
image_formation.py - تشکیل تصویر از منابع مختلف (فایل، Base64، URL، RTSP، دوربین USB)
"""

import cv2
import base64
import numpy as np
from fastapi import UploadFile
from typing import Optional, Union
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)


class ImageFormation:
    @staticmethod
    async def from_upload(file: UploadFile) -> np.ndarray:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("فرمت تصویر نامعتبر است")
        return img

    @staticmethod
    def from_base64(base64_str: str) -> np.ndarray:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Base64 معتبر نیست")
        return img

    @staticmethod
    def from_path(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"فایل {path} یافت نشد")
        return img

    @staticmethod
    async def from_url(url: str, timeout: int = 10) -> np.ndarray:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise Exception(f"خطا در دریافت تصویر: {resp.status}")
                data = await resp.read()
                np_arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("تصویر دریافتی نامعتبر است")
                return img

    @staticmethod
    async def from_rtsp(rtsp_url: str, timeout: int = 5) -> np.ndarray:
        loop = asyncio.get_event_loop()
        def _capture():
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise Exception("خطا در اتصال به دوربین RTSP")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise Exception("عدم دریافت فریم از دوربین")
            return frame
        return await loop.run_in_executor(None, _capture)

    @staticmethod
    def from_array(array: np.ndarray) -> np.ndarray:
        if array is None or array.size == 0:
            raise ValueError("آرایه تصویر خالی است")
        if len(array.shape) not in [2, 3]:
            raise ValueError("بعد آرایه تصویر نامعتبر است")
        return array

    @staticmethod
    def to_bytes(img: np.ndarray, format: str = '.jpg') -> bytes:
        _, buffer = cv2.imencode(format, img)
        return buffer.tobytes()

    @staticmethod
    def to_base64(img: np.ndarray, format: str = '.jpg') -> str:
        return base64.b64encode(ImageFormation.to_bytes(img, format)).decode('utf-8')