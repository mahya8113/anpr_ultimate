"""
live.py - WebSocket برای پخش زنده دوربین‌ها و تشخیص لحظه‌ای پلاک
"""

import asyncio
import base64
import cv2
import json
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import logging

from core.redis_client import redis_client
from core.security import verify_websocket_token
from modules.detection import ObjectDetector
from modules.advanced_ocr import AdvancedOCR
from modules.multi_tracking import MultiTracker
from utils.camera_manager import CameraConnector

logger = logging.getLogger(__name__)

# ==================== مدیریت اتصالات ====================
class ConnectionManager:
    """مدیریت اتصالات وب‌سوکت به دوربین‌ها"""
    def init(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.detector = ObjectDetector()
        self.ocr = AdvancedOCR()
        self.tracker = MultiTracker()
        self.camera_streams: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, camera_id: str, token: str):
        """پذیرش اتصال جدید و تأیید توکن"""
        user = await verify_websocket_token(token)
        if not user:
            await websocket.close(code=1008, reason="توکن نامعتبر است")
            return None
        
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = set()
        self.active_connections[camera_id].add(websocket)
        
        logger.info(f"WebSocket connected: camera={camera_id}, user={user['id']}")
        return user
    
    def disconnect(self, websocket: WebSocket, camera_id: str):
        """قطع اتصال یک کلاینت"""
        if camera_id in self.active_connections:
            self.active_connections[camera_id].discard(websocket)
            if not self.active_connections[camera_id]:
                del self.active_connections[camera_id]
                # توقف استریم اگر بیننده‌ای باقی نمانده باشد
                if camera_id in self.camera_streams:
                    self.camera_streams[camera_id].cancel()
                    del self.camera_streams[camera_id]
        logger.info(f"WebSocket disconnected: camera={camera_id}")
    
    async def broadcast_frame(self, camera_id: str, frame_b64: str, analysis: dict):
        """ارسال فریم و تحلیل به همه بینندگان یک دوربین"""
        if camera_id not in self.active_connections:
            return
        
        message = json.dumps({
            "type": "frame",
            "camera_id": camera_id,
            "frame": frame_b64,
            "analysis": analysis,
            "timestamp": asyncio.get_event_loop().time()
        })
        disconnected = []
        for ws in self.active_connections[camera_id]:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws, camera_id)
    
    async def start_camera_stream(self, camera_id: str, stream_url: str, fps: int = 15):
        """پس‌زمینه: دریافت فریم از دوربین و تشخیص پلاک"""
        if camera_id in self.camera_streams:
            return  # قبلاً در حال اجراست
        
        async def _stream_worker():
            frame_interval = 1.0 / max(fps, 1)
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"Cannot open camera stream: {camera_id} - {stream_url}")
                return
            
            frame_count = 0
            try:
                while True:
                    # اگر بیننده‌ای وجود نداشت، استریم را متوقف کن
                    if camera_id not in self.active_connections:
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        # تلاش مجدد برای اتصال
                        await asyncio.sleep(1)
                        cap.release()
                        cap = cv2.VideoCapture(stream_url)
                        continue
                    
                    frame_count += 1
                    # پردازش هر N فریم (کاهش بار)
                    analysis = {"num_plates": 0, "plates": []}
                    if frame_count % 10 == 0:
                        plates = self.detector.detect_plates(frame)
                        results = []
                        for plate in plates:
                            x1, y1, x2, y2 = map(int, plate['bbox'])
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                text, conf, _ = self.ocr.read_plate(crop)
                                results.append({
                                    "plate": text,
                                    "confidence": plate['confidence'],
                                    "ocr_confidence": conf,
                                    "bbox": plate['bbox']
                                })
                        analysis = {"num_plates": len(results), "plates": results}
                    
                    # تبدیل فریم به base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await self.broadcast_frame(camera_id, frame_b64, analysis)
                    await asyncio.sleep(frame_interval)
            finally:
                cap.release()
                if camera_id in self.camera_streams:
                    del self.camera_streams[camera_id]
        
        task = asyncio.create_task(_stream_worker())
        self.camera_streams[camera_id] = task


manager = ConnectionManager()


# ==================== اندپوینت WebSocket ====================
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

router = APIRouter()


@router.websocket("/live/{camera_id}")
async def websocket_live(
    websocket: WebSocket,
    camera_id: str,
    token: str = Query(...),
    stream_url: str = Query(None)
):
    """
    WebSocket برای پخش زنده دوربین
    آدرس نمونه: ws://localhost:8000/ws/live/cam_123?token=xxx&stream_url=rtsp://...
    """
    # اعتبارسنجی توکن
    user = await manager.connect(websocket, camera_id, token)
    if user is None:
        return
    
    # دریافت آدرس استریم (از دیتابیس یا از پارامتر)
    if not stream_url:
        # در اینجا می‌توان از دیتابیس بر اساس camera_id آدرس را گرفت
        # برای نمونه از یک مقدار پیش‌فرض استفاده می‌کنیم
        stream_url = f"rtsp://example.com/{camera_id}"
    
    # شروع استریم دوربین در پس‌زمینه
    asyncio.create_task(manager.start_camera_stream(camera_id, stream_url))
    
    try:
        # نگهداری اتصال زنده (کلاینت ممکن است پیام کنترلی بفرستد)
        while True:
            data = await websocket.receive_text()
            # می‌توان پیام‌های کنترلی مثل تغییر کیفیت، تغییر دوربین و ... را پردازش کرد
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)
        logger.info(f"WebSocket disconnected: camera={camera_id}")