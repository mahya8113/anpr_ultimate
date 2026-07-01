"""
live.py - صفحه پخش زنده دوربین‌ها و نظارت لحظه‌ای
قابلیت‌ها: نمایش استریم زنده، تشخیص لحظه‌ای پلاک، نمایش نتایج، هشدارها
"""

import streamlit as st
import cv2
import numpy as np
import requests
import json
import asyncio
import base64
from datetime import datetime
from PIL import Image
import io
import plotly.graph_objects as go

# تنظیمات صفحه
st.set_page_config(
    page_title="پخش زنده | سامانه تشخیص پلاک",
    page_icon="📡",
    layout="wide"
)

# CSS برای راست‌چین و فارسی
st.markdown("""
<style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stButton button {
        width: 100%;
    }
    .cam-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .plate-result {
        background: #1e3c72;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 18px;
    }
    .alert-box {
        background: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box {
        background: #00cc88;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# عنوان صفحه
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3c72, #2a5298); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1>📡 پایش زنده دوربین‌ها</h1>
    <p>تشخیص لحظه‌ای پلاک خودروها - نظارت بر تردد - هشدارهای آنی</p>
</div>
""", unsafe_allow_html=True)

# بررسی احراز هویت
if "token" not in st.session_state:
    st.error("❌ لطفاً ابتدا وارد سامانه شوید")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}
API_BASE = st.session_state.get("api_base", "http://localhost:8000")


# ==================== توابع کمکی ====================

def get_cameras():
    """دریافت لیست دوربین‌های سازمان"""
    try:
        resp = requests.get(f"{API_BASE}/admin/cameras", headers=headers)
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "خطا در دریافت دوربین‌ها")
    except Exception as e:
        return None, str(e)


def detect_plate(image_bytes):
    """ارسال تصویر به API برای تشخیص پلاک"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        resp = requests.post(f"{API_BASE}/detect", headers=headers, files=files)
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "خطا در تشخیص")
    except Exception as e:
        return None, str(e)


def get_recent_detections(limit=20):
    """دریافت آخرین تشخیص‌ها"""
    try:
        resp = requests.get(f"{API_BASE}/admin/dashboard/recent-detections", headers=headers)
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "خطا")
    except Exception as e:
        return None, str(e)


# ==================== انتخاب دوربین ====================

st.sidebar.markdown("## 🎥 انتخاب دوربین")

cameras, error = get_cameras()
if error:
    st.sidebar.error(f"خطا: {error}")
    cameras = []

if cameras:
    camera_names = [f"{cam['name']} ({cam.get('location', 'نامشخص')})" for cam in cameras]
    selected_cam_name = st.sidebar.selectbox("دوربین مورد نظر", camera_names)
    selected_cam = cameras[camera_names.index(selected_cam_name)]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    اطلاعات دوربین:
    - آیدی: {selected_cam.get('id')}
    - وضعیت: {'🟢 فعال' if selected_cam.get('is_active') else '🔴 غیرفعال'}
    - نوع استریم: {selected_cam.get('stream_type', 'rtsp')}
    """)
else:
    st.sidebar.warning("هیچ دوربینی یافت نشد. لطفاً ابتدا دوربین اضافه کنید.")
    selected_cam = None


# ==================== تب‌های اصلی ====================

tab1, tab2, tab3 = st.tabs(["📡 پخش زنده", "📋 آخرین تشخیص‌ها", "📊 آمار لحظه‌ای"])


# ==================== تب 1: پخش زنده ====================
with tab1:
    if selected_cam:
        st.markdown(f"### 🎥 {selected_cam['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # نمایش استریم زنده
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # دکمه شروع/توقف
            if "streaming" not in st.session_state:
                st.session_state.streaming = False
            
            col_start, col_stop, col_capture = st.columns(3)
            
            with col_start:
                if st.button("▶️ شروع پخش", use_container_width=True):
                    st.session_state.streaming = True
            
            with col_stop:
                if st.button("⏹️ توقف پخش", use_container_width=True):
                    st.session_state.streaming = False
            
            with col_capture:
                capture_mode = st.checkbox("📸 حالت عکس (Single Frame)")
            
            st.markdown("---")
            
            # پردازش و نمایش فریم‌ها
            if st.session_state.streaming and selected_cam:
                stream_url = selected_cam['stream_url']
                
                if capture_mode:
                    # حالت تک فریم
                    if st.button("📸 ثبت و تشخیص"):
                        with st.spinner("در حال تشخیص..."):
                            cap = cv2.VideoCapture(stream_url)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                _, buffer = cv2.imencode('.jpg', frame)
                                result, error = detect_plate(buffer.tobytes())
                                
                                if result and result.get('plates'):
                                    st.success(f"✅ پلاک شناسایی شد: {result['plates']}")
                                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                                else:
                                    st.warning("هیچ پلاکی تشخیص داده نشد")
                                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                            else:
                                st.error("خطا در دریافت تصویر از دوربین")
                else:
                    # حالت استریم زنده (ساده شده - در نسخه واقعی از WebSocket استفاده می‌شود)
                    status_placeholder.info("🔄 در حال اتصال به استریم دوربین...")
                    
                    cap = cv2.VideoCapture(stream_url)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = 0
                    
                    while st.session_state.streaming:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.error("قطع ارتباط با دوربین")
                            break
                        
                        frame_count += 1
                        
                        # هر N فریم یکبار تشخیص انجام شود (برای کاهش بار)
                        if frame_count % 30 == 0:
                            _, buffer = cv2.imencode('.jpg', frame)
                            result, error = detect_plate(buffer.tobytes())
                            
                            if result and result.get('plates'):
                                status_placeholder.success(f"✅ پلاک: {result['plates']}")
                            else:
                                status_placeholder.info("در حال پایش...")
                        
                        # نمایش فریم
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        cap.release()
        
        with col2:
            st.markdown("### 📝 نتایج تشخیص لحظه‌ای")
            
            # نمایش نتایج در یک کانتینر
            results_container = st.empty()
            
            # نمونه نتایج (در نسخه واقعی از WebSocket دریافت می‌شود)
            sample_results = [
                {"plate": "۱۲۳۴۵۶۷", "confidence": 0.95, "time": datetime.now().strftime("%H:%M:%S")},
                {"plate": "۸۹۱۲۳۴۵", "confidence": 0.87, "time": (datetime.now().replace(second=datetime.now().second-5)).strftime("%H:%M:%S")},
            ]
            
            for res in sample_results:
                results_container.markdown(f"""
                <div class="plate-result">
                    🚗 پلاک: {res['plate']} | اطمینان: {res['confidence']*100:.0f}% | زمان: {res['time']}
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.warning("لطفاً ابتدا یک دوربین از منوی کناری انتخاب کنید")


# ==================== تب 2: آخرین تشخیص‌ها ====================

with tab2:
    st.markdown("### 📋 آخرین پلاک‌های شناسایی شده")
    
    detections, error = get_recent_detections(limit=20)
    
    if error:
        st.error(f"خطا: {error}")
    elif detections:
        # نمایش به صورت جدول
        import pandas as pd
        df = pd.DataFrame(detections)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.dataframe(df, use_container_width=True)
        
        # نمایش جزئیات تشخیص انتخاب شده
        if len(detections) > 0:
            selected_idx = st.selectbox("انتخاب تشخیص برای مشاهده جزئیات", range(len(detections)))
            if selected_idx is not None:
                st.json(detections[selected_idx])
    else:
        st.info("هیچ تشخیصی یافت نشد")


# ==================== تب 3: آمار لحظه‌ای ====================

with tab3:
    st.markdown("### 📊 آمار لحظه‌ای تردد")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # دریافت آمار از API
    stats, error = get_recent_detections(limit=100)
    
    if stats:
        # محاسبه آمار
        total_today = len([s for s in stats if datetime.fromisoformat(s.get("created_at", "2000-01-01")).date() == datetime.now().date()])
        total_hour = len([s for s in stats if (datetime.now() - datetime.fromisoformat(s.get("created_at", "2000-01-01"))).seconds < 3600])
        
        with col1:
            st.metric("🚗 تردد امروز", total_today)
        with col2:
            st.metric("⏱️ تردد یک ساعت اخیر", total_hour)
        with col3:
            st.metric("🎯 میانگین اطمینان", f"{np.mean([s.get('confidence', 0) for s in stats])*100:.0f}%")
        with col4:
            st.metric("🕐 آخرین تشخیص", "دقایقی پیش")
        
        st.markdown("---")
        
        # نمودار تردد در ساعات مختلف
        if len(stats) > 0:
            import pandas as pd
            df = pd.DataFrame(stats)
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
                df["hour"] = df["created_at"].dt.hour
                hourly = df.groupby("hour").size().reset_index(name="count")
                
                fig = go.Figure(data=[
                    go.Bar(x=hourly["hour"], y=hourly["count"], marker_color="#1e3c72")
                ])
                fig.update_layout(
                    title="پراکندگی تردد در ساعات مختلف",
                    xaxis_title="ساعت",
                    yaxis_title="تعداد تردد",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("داده‌ای برای نمایش وجود ندارد")


# ==================== هشدارها ====================

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚠️ هشدارهای لحظه‌ای")

# نمایش هشدارهای اخیر
alerts = [
    {"type": "warning", "message": "پلاک نامعتبر - ورود ممنوع", "time": "2 دقیقه پیش"},
    {"type": "info", "message": "خودروی مشکوک - سرعت غیرمجاز", "time": "5 دقیقه پیش"},
]
for alert in alerts:
    color = "#ffcc00" if alert["type"] == "warning" else "#00cc88"
    st.sidebar.markdown(f"""
    <div style="background: {color}; padding: 8px; border-radius: 5px; margin: 5px 0; font-size: 12px;">
        ⚠️ {alert['message']}<br>
        <small>{alert['time']}</small>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"آخرین بروزرسانی: {datetime.now().strftime('%H:%M:%S')}")