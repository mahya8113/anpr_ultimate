"""
test_api.py
تست‌های واحد و یکپارچه برای APIهای سامانه تشخیص پلاک
"""

import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import base64

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.main import app

client = TestClient(app)


# ==================== تست‌های سلامت ====================
class TestHealth:
    """تست‌های مربوط به سلامت سامانه"""
    
    def test_root_endpoint(self):
        """تست endpoint اصلی"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self):
        """تست endpoint سلامت"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_metrics_endpoint(self):
        """تست endpoint متریک‌ها"""
        response = client.get("/metrics")
        assert response.status_code == 200


# ==================== تست‌های احراز هویت ====================
class TestAuth:
    """تست‌های مربوط به احراز هویت"""
    
    def test_register_user(self):
        """تست ثبت نام کاربر"""
        user_data = {
            "email": "test@example.com",
            "password": "test123456",
            "full_name": "کاربر تست",
            "org_name": "سازمان تست"
        }
        response = client.post("/auth/register", json=user_data)
        
        if response.status_code == 201:
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
        elif response.status_code == 400:
            assert "قبلاً ثبت شده" in response.json()["detail"]
    
    def test_login_user(self):
        """تست ورود کاربر"""
        login_data = {
            "email": "test@example.com",
            "password": "test123456"
        }
        response = client.post("/auth/login", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data
        else:
            assert response.status_code in [401, 400]
    
    def test_login_invalid_credentials(self):
        """تست ورود با اطلاعات نادرست"""
        login_data = {
            "email": "invalid@example.com",
            "password": "wrongpassword"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_refresh_token(self):
        """تست بازسازی توکن"""
        # ابتدا لاگین کنید
        login_data = {
            "email": "test@example.com",
            "password": "test123456"
        }
        login_resp = client.post("/auth/login", json=login_data)
        
        if login_resp.status_code == 200:
            refresh_token = login_resp.json()["refresh_token"]
            response = client.post(f"/auth/refresh?refresh_token={refresh_token}")
            assert response.status_code == 200
            assert "access_token" in response.json()


# ==================== تست‌های تشخیص پلاک ====================
class TestDetection:
    """تست‌های مربوط به تشخیص پلاک"""
    
    @pytest.fixture
    def sample_image_base64(self):
        """ایجاد تصویر نمونه به صورت base64"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return img_b64
    
    def test_detect_from_base64(self, sample_image_base64):
        """تست تشخیص پلاک از تصویر base64"""
        payload = {"image_base64": sample_image_base64}
        response = client.post("/detect", json=payload)
        assert response.status_code in [200, 422]  # 422 اگر تصویر معتبر نباشد
    
    def test_detect_from_file_upload(self):
        """تست تشخیص پلاک از آپلود فایل"""
        # ایجاد تصویر موقت
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', img)
        
        files = {"file": ("test.jpg", buffer.tobytes(), "image/jpeg")}
        response = client.post("/detect/upload", files=files)
        
        assert response.status_code in [200, 422]
    
    def test_detect_invalid_image(self):
        """تست تشخیص با تصویر نامعتبر"""
        payload = {"image_base64": "invalid_base64"}
        response = client.post("/detect", json=payload)
        assert response.status_code == 400
    
    def test_detect_empty_image(self):
        """تست تشخیص با تصویر خالی"""
        payload = {"image_base64": ""}
        response = client.post("/detect", json=payload)
        assert response.status_code == 400


# ==================== تست‌های دوربین ====================
class TestCamera:
    """تست‌های مربوط به مدیریت دوربین‌ها"""
    
    @pytest.fixture
    def auth_headers(self):
        """ایجاد هدر احراز هویت"""
        return {"Authorization": "Bearer test_token"}
    
    def test_list_cameras(self, auth_headers):
        """تست دریافت لیست دوربین‌ها"""
        response = client.get("/cameras", headers=auth_headers)
        assert response.status_code in [200, 401]
    
    def test_add_camera(self, auth_headers):
        """تست افزودن دوربین جدید"""
        camera_data = {
            "name": "دوربین تست",
            "stream_url": "rtsp://test.stream",
            "stream_type": "rtsp",
            "location": "ورودی جنوبی"
        }
        response = client.post("/cameras", json=camera_data, headers=auth_headers)
        assert response.status_code in [200, 201, 401, 422]


# ==================== تست‌های گزارش ====================
class TestReports:
    """تست‌های مربوط به گزارش‌گیری"""
    
    def test_get_detections_report(self):
        """تست دریافت گزارش تشخیص‌ها"""
        params = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
        response = client.get("/reports/detections", params=params)
        assert response.status_code in [200, 401]
    
    def test_download_report_pdf(self):
        """تست دانلود گزارش PDF"""
        params = {
            "format": "pdf",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
        response = client.get("/reports/download", params=params)
        assert response.status_code in [200, 401]
    
    def test_plate_statistics(self):
        """تست آمار یک پلاک خاص"""
        response = client.get("/reports/plate-stats/1234567")
        assert response.status_code in [200, 401, 404]
    
    def test_camera_performance(self):
        """تست عملکرد دوربین"""
        response = client.get("/reports/camera-performance/1")
        assert response.status_code in [200, 401, 404]


# ==================== تست‌های مدیریت ====================
class TestAdmin:
    """تست‌های مربوط به پنل مدیریت"""
    
    @pytest.fixture
    def admin_headers(self):
        """ایجاد هدر ادمین"""
        return {"Authorization": "Bearer admin_token"}
    
    def test_list_organizations(self, admin_headers):
        """تست دریافت لیست سازمان‌ها"""
        response = client.get("/admin/organizations", headers=admin_headers)
        assert response.status_code in [200, 401]
    
    def test_create_organization(self, admin_headers):
        """تست ایجاد سازمان جدید"""
        org_data = {
            "name": "سازمان جدید",
            "tier": "standard",
            "max_cameras": 5
        }
        response = client.post("/admin/organizations", json=org_data, headers=admin_headers)
        assert response.status_code in [200, 201, 401, 422]
    
    def test_dashboard_stats(self, admin_headers):
        """تست آمار داشبورد"""
        response = client.get("/admin/dashboard/stats", headers=admin_headers)
        assert response.status_code in [200, 401]
    
    def test_system_health(self, admin_headers):
        """تست سلامت سیستم"""
        response = client.get("/admin/health", headers=admin_headers)
        assert response.status_code in [200, 401]


# ==================== تست‌های وب‌سوکت ====================
@pytest.mark.asyncio
class TestWebSocket:
    """تست‌های مربوط به وب‌سوکت"""
    
    async def test_websocket_connection(self):
        """تست اتصال وب‌سوکت"""
        # این تست نیاز به websocket client دارد
        pass


# ==================== تست‌های سرعت و بار ====================
class TestPerformance:
    """تست‌های عملکرد و سرعت"""
    
    def test_detection_response_time(self):
        """تست زمان پاسخگویی تشخیص پلاک"""
        import time
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        start = time.time()
        response = client.post("/detect", json={"image_base64": img_b64})
        end = time.time()
        
        if response.status_code == 200:
            assert (end - start) < 2.0  # کمتر از 2 ثانیه
    
    def test_concurrent_requests(self):
        """تست درخواست‌های همزمان"""
        import concurrent.futures
        
        def make_request():
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _, buffer = cv2.imencode('.jpg', img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            return client.post("/detect", json={"image_base64": img_b64})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                assert response.status_code in [200, 422, 429]


# ==================== اجرای تست‌ها ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])