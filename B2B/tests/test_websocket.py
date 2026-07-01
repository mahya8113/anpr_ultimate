"""
test_websocket.py
تست‌های مربوط به WebSocket برای پخش زنده دوربین‌ها و ارتباط بلادرنگ
شامل: اتصال، ارسال فریم، قطع اتصال، مدیریت چند کلاینت
"""

import pytest
import asyncio
import json
import base64
import numpy as np
import cv2
from fastapi import WebSocket
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
import websockets
import threading
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.websocket.live import ConnectionManager
from backend.main import app


# ==================== Fixtures ====================
@pytest.fixture
def sample_frame():
    """ایجاد فریم نمونه برای تست وب‌سوکت"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return frame_b64


@pytest.fixture
def mock_websocket():
    """Mock WebSocket برای تست"""
    mock = AsyncMock(spec=WebSocket)
    mock.accept = AsyncMock()
    mock.send_text = AsyncMock()
    mock.send_json = AsyncMock()
    mock.receive_text = AsyncMock()
    mock.receive_json = AsyncMock()
    mock.close = AsyncMock()
    return mock


# ==================== تست‌های ConnectionManager ====================
class TestConnectionManager:
    """تست‌های کلاس مدیریت اتصالات WebSocket"""
    
    def test_init_connection_manager(self):
        """تست مقداردهی اولیه ConnectionManager"""
        manager = ConnectionManager()
        
        assert manager.active_connections == {}
        assert manager.detector is not None
        assert manager.ocr is not None
        assert manager.tracker is not None
    
    @pytest.mark.asyncio
    async def test_connect(self, mock_websocket):
        """تست اتصال کلاینت به WebSocket"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        await manager.connect(mock_websocket, camera_id)
        
        mock_websocket.accept.assert_called_once()
        assert camera_id in manager.active_connections
        assert mock_websocket in manager.active_connections[camera_id]
    
    @pytest.mark.asyncio
    async def test_connect_multiple_clients(self, mock_websocket):
        """تست اتصال چند کلاینت به یک دوربین"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        mock_websocket2 = AsyncMock(spec=WebSocket)
        
        await manager.connect(mock_websocket, camera_id)
        await manager.connect(mock_websocket2, camera_id)
        
        assert len(manager.active_connections[camera_id]) == 2
        assert mock_websocket in manager.active_connections[camera_id]
        assert mock_websocket2 in manager.active_connections[camera_id]
    
    @pytest.mark.asyncio
    async def test_connect_different_cameras(self, mock_websocket):
        """تست اتصال به دوربین‌های مختلف"""
        manager = ConnectionManager()
        
        await manager.connect(mock_websocket, "camera_1")
        await manager.connect(mock_websocket, "camera_2")
        
        assert "camera_1" in manager.active_connections
        assert "camera_2" in manager.active_connections
        assert len(manager.active_connections) == 2
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_websocket):
        """تست断开 اتصال کلاینت"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        await manager.connect(mock_websocket, camera_id)
        manager.disconnect(mock_websocket, camera_id)
        
        assert mock_websocket not in manager.active_connections.get(camera_id, set())
    
    @pytest.mark.asyncio
    async def test_disconnect_last_client(self, mock_websocket):
        """تست断开 اتصال آخرین کلاینت"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        await manager.connect(mock_websocket, camera_id)
        manager.disconnect(mock_websocket, camera_id)
        
        assert camera_id not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_frame(self, mock_websocket, sample_frame):
        """تست ارسال فریم به همه کلاینت‌ها"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        await manager.connect(mock_websocket, camera_id)
        
        analysis = {"num_plates": 1, "plates": ["1234567"]}
        await manager.broadcast_frame(camera_id, sample_frame, analysis)
        
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        sent_data = json.loads(call_args)
        
        assert sent_data["type"] == "frame"
        assert sent_data["camera_id"] == camera_id
        assert sent_data["frame"] == sample_frame
        assert sent_data["analysis"] == analysis
    
    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self, mock_websocket, sample_frame):
        """تست ارسال فریم بدون کلاینت متصل"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        # بدون اتصال کلاینت
        await manager.broadcast_frame(camera_id, sample_frame, {})
        
        mock_websocket.send_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_send_alert(self, mock_websocket):
        """تست ارسال هشدار به کلاینت‌ها"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        await manager.connect(mock_websocket, camera_id)
        
        alert = {"type": "anomaly", "message": "حرکت خلاف جهت", "plate": "1234567"}
        await manager.send_alert(camera_id, alert)
        
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        sent_data = json.loads(call_args)
        
        assert sent_data["type"] == "alert"
        assert sent_data["alert"] == alert


# ==================== تست‌های WebSocket Endpoint ====================
class TestWebSocketEndpoint:
    """تست‌های endpoint WebSocket"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_success(self):
        """تست موفقیت‌آمیز اتصال WebSocket"""
        # این تست نیاز به راه‌اندازی سرور تست دارد
        pass
    
    @pytest.mark.asyncio
    async def test_websocket_connection_invalid_token(self):
        """تست اتصال با توکن نامعتبر"""
        # این تست نیاز به راه‌اندازی سرور تست دارد
        pass
    
    @pytest.mark.asyncio
    async def test_websocket_connection_missing_token(self):
        """تست اتصال بدون توکن"""
        # این تست نیاز به راه‌اندازی سرور تست دارد
        pass


# ==================== تست‌های پردازش فریم ====================
class TestFrameProcessing:
    """تست‌های پردازش فریم در وب‌سوکت"""
    
    @pytest.mark.asyncio
    async def test_process_frame(self):
        """تست پردازش فریم و تشخیص پلاک"""
        from backend.modules.detection import ObjectDetector
        
        detector = ObjectDetector()
        
        # ایجاد فریم نمونه
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # تشخیص (ممکن است چیزی پیدا نکند)
        plates = detector.detect(frame)
        
        assert isinstance(plates, list)
    
    @pytest.mark.asyncio
    async def test_frame_to_base64(self, sample_frame):
        """تست تبدیل فریم به base64"""
        assert isinstance(sample_frame, str)
        assert len(sample_frame) > 0


# ==================== تست‌های بار و عملکرد ====================
class TestWebSocketPerformance:
    """تست‌های عملکرد وب‌سوکت با بار بالا"""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_connections(self):
        """تست اتصال همزمان چندین کلاینت"""
        manager = ConnectionManager()
        connections = []
        
        for i in range(10):
            mock = AsyncMock(spec=WebSocket)
            await manager.connect(mock, f"camera_{i % 2}")
            connections.append(mock)
        
        assert len(manager.active_connections) == 2  # دو دوربین
        assert manager.active_connections.get("camera_0", set()) is not None
    
    @pytest.mark.asyncio
    async def test_rapid_frame_broadcast(self, mock_websocket, sample_frame):
        """تست ارسال سریع فریم‌های متوالی"""
        manager = ConnectionManager()
        camera_id = "test_camera"
        
        await manager.connect(mock_websocket, camera_id)
        
        # ارسال 100 فریم متوالی
        for i in range(100):
            await manager.broadcast_frame(camera_id, sample_frame, {"frame": i})
        
        assert mock_websocket.send_text.call_count == 100


# ==================== تست‌های بازیابی خطا ====================
class TestWebSocketErrorRecovery:
    """تست‌های بازیابی خطا در وب‌سوکت"""
    
    @pytest.mark.asyncio
    async def test_connection_recovery(self):
        """تست بازیابی اتصال قطع شده"""
        manager = ConnectionManager()
        mock = AsyncMock(spec=WebSocket)
        
        # شبیه‌سازی خطا در ارسال
        mock.send_text.side_effect = Exception("Connection lost")
        
        await manager.connect(mock, "test_camera")
        
        # تلاش برای ارسال
        await manager.broadcast_frame("test_camera", "frame_data", {})
        
        # باید اتصال قطع شده باشد
        assert "test_camera" not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self):
        """تست تایم اوت heartbeat"""
        # این تست نیاز به پیاده‌سازی heartbeat دارد
        pass


# ==================== تست‌های یکپارچگی با فرانت‌اند ====================
class TestFrontendIntegration:
    """تست‌های یکپارچگی با فرانت‌اند استریملیت"""
    
    @pytest.mark.asyncio
    async def test_live_page_websocket(self):
        """تست صفحه پخش زنده و اتصال WebSocket"""
        # این تست نیاز به اجرای فرانت‌اند دارد
        pass


# ==================== اجرای تست‌ها ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])