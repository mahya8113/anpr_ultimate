"""
camera_manager.py - مدیریت اتصال به دوربین‌های مختلف
پشتیبانی از: RTSP, HTTP, USB, V4L2, ONVIF
"""

import cv2
import asyncio
import aiohttp
import numpy as np
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CameraConnector:
    """کلاس اصلی مدیریت اتصال به دوربین‌ها"""
    
    @staticmethod
    async def get_frame_rtsp(rtsp_url: str, timeout: int = 5) -> Optional[np.ndarray]:
        """
        دریافت فریم از دوربین RTSP
        
        Args:
            rtsp_url: آدرس RTSP (مثال: rtsp://192.168.1.100:554/stream)
            timeout: زمان انتظار به ثانیه
        
        Returns:
            فریم تصویر یا None در صورت خطا
        """
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.error(f"خطا در اتصال به RTSP: {rtsp_url}")
                return None
            
            # تنظیم timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            else:
                logger.warning(f"عدم دریافت فریم از {rtsp_url}")
                return None
                
        except Exception as e:
            logger.error(f"خطا در دریافت فریم RTSP: {e}")
            return None
    
    @staticmethod
    async def get_frame_http_mjpeg(http_url: str, timeout: int = 5) -> Optional[np.ndarray]:
        """
        دریافت فریم از دوربین HTTP MJPEG
        
        Args:
            http_url: آدرس HTTP (مثال: http://192.168.1.100/video)
            timeout: زمان انتظار به ثانیه
        
        Returns:
            فریم تصویر یا None در صورت خطا
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(http_url, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        np_arr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        return frame
                    else:
                        logger.error(f"خطا در HTTP: {resp.status}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout در اتصال به {http_url}")
            return None
        except Exception as e:
            logger.error(f"خطا در دریافت فریم HTTP: {e}")
            return None
    
    @staticmethod
    def get_frame_usb(camera_id: int = 0) -> Optional[np.ndarray]:
        """
        دریافت فریم از دوربین USB یا داخلی لپ‌تاپ
        
        Args:
            camera_id: شناسه دوربین (0, 1, 2, ...)
        
        Returns:
            فریم تصویر یا None در صورت خطا
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"خطا در اتصال به دوربین USB {camera_id}")
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"خطا در دریافت فریم USB: {e}")
            return None
    
    @staticmethod
    def get_frame_v4l2(device_path: str = "/dev/video0") -> Optional[np.ndarray]:
        """
        دریافت فریم از دستگاه V4L2 (کارت‌های کپچر آنالوگ)
        
        Args:
            device_path: مسیر دستگاه (مثال: /dev/video0)
        
        Returns:
            فریم تصویر یا None در صورت خطا
        """
        try:
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if not cap.isOpened():
                logger.error(f"خطا در اتصال به V4L2 {device_path}")
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"خطا در دریافت فریم V4L2: {e}")
            return None
    
    @staticmethod
    async def discover_onvif(
        ip: str, 
        port: int = 80, 
        username: Optional[str] = None, 
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        کشف دوربین ONVIF و دریافت پروفایل‌ها
        
        Args:
            ip: آدرس IP دوربین
            port: پورت ONVIF
            username: نام کاربری (اختیاری)
            password: رمز عبور (اختیاری)
        
        Returns:
            دیکشنری شامل اطلاعات دوربین و آدرس RTSP
        """
        try:
            from onvif import ONVIFCamera
            
            # اتصال به دوربین ONVIF
            camera = ONVIFCamera(ip, port, username, password)
            
            # دریافت سرویس مدیا
            media = camera.create_media_service()
            
            # دریافت پروفایل‌ها
            profiles = media.GetProfiles()
            
            rtsp_urls = []
            for profile in profiles:
                try:
                    stream_uri = media.GetStreamUri({
                        'StreamSetup': {
                            'Stream': 'RTP-Unicast',
                            'Transport': 'RTSP'
                        },
                        'ProfileToken': profile.token
                    })
                    rtsp_urls.append(stream_uri.Uri)
                except Exception as e:
                    logger.warning(f"خطا در دریافت RTSP برای پروفایل {profile.token}: {e}")
            
            return {
                "status": "success",
                "device_info": {
                    "manufacturer": camera.GetDeviceInformation().get('Manufacturer', 'Unknown'),
                    "model": camera.GetDeviceInformation().get('Model', 'Unknown'),
                    "firmware": camera.GetDeviceInformation().get('FirmwareVersion', 'Unknown')
                },
                "profiles_count": len(profiles),
                "rtsp_urls": rtsp_urls
            }
            
        except ImportError:
            logger.warning("کتابخانه ONVIF نصب نیست. نصب: pip install onvif-zeep")
            return {"status": "error", "message": "ONVIF library not installed"}
        except Exception as e:
            logger.error(f"خطا در کشف ONVIF: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def get_frame_from_uri(uri: str) -> Optional[np.ndarray]:
        """
        دریافت فریم از هر نوع URI به صورت خودکار
        
        Args:
            uri: آدرس دوربین (rtsp://, http://, usb://, v4l2://)
        
        Returns:
            فریم تصویر یا None در صورت خطا
        """
        parsed = urlparse(uri)
        
        if parsed.scheme == 'rtsp':
            return await CameraConnector.get_frame_rtsp(uri)
        
        elif parsed.scheme in ('http', 'https'):
            return await CameraConnector.get_frame_http_mjpeg(uri)
        
        elif parsed.scheme == 'usb':
            camera_id = int(parsed.netloc) if parsed.netloc else 0
            return CameraConnector.get_frame_usb(camera_id)
        
        elif parsed.scheme == 'v4l2':
            device_path = parsed.netloc or '/dev/video0'
            return CameraConnector.get_frame_v4l2(device_path)
        
        else:
            logger.error(f"پروتکل ناشناخته: {parsed.scheme}")
            return None
    
    @staticmethod
    def test_connection(uri: str) -> Tuple[bool, str]:
        """
        تست اتصال به دوربین
        
        Args:
            uri: آدرس دوربین
        
        Returns:
            (موفقیت, پیام)
        """
        try:
            import asyncio
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            frame = loop.run_until_complete(
                CameraConnector.get_frame_from_uri(uri)
            )
            loop.close()
            
            if frame is not None:
                return True, "اتصال با موفقیت برقرار شد"
            else:
                return False, "عدم دریافت فریم از دوربین"
                
        except Exception as e:
            return False, f"خطا در اتصال: {str(e)}"


class CameraStream:
    """مدیریت استریم زنده از دوربین"""
    
    def init(self, uri: str, fps: int = 30):
        self.uri = uri
        self.fps = fps
        self.cap = None
        self.is_running = False
        
    async def start(self):
        """شروع استریم"""
        parsed = urlparse(self.uri)
        
        if parsed.scheme == 'rtsp':
            self.cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        elif parsed.scheme == 'usb':
            camera_id = int(parsed.netloc) if parsed.netloc else 0
            self.cap = cv2.VideoCapture(camera_id)
        else:
            self.cap = cv2.VideoCapture(self.uri)
        
        if not self.cap.isOpened():
            raise Exception(f"خطا در باز کردن دوربین: {self.uri}")
        
        self.is_running = True
        logger.info(f"استریم دوربین {self.uri} شروع شد")
    
    async def read_frame(self) -> Optional[np.ndarray]:
        """خواندن یک فریم"""
        if not self.is_running or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    async def stop(self):
        """توقف استریم"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info(f"استریم دوربین {self.uri} متوقف شد")
    
    async def aenter(self):
        await self.start()
        return self
    
    async def aexit(self, exc_type, exc_val, exc_tb):
        await self.stop()