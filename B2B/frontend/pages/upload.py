"""
upload.py - صفحه آپلود و پردازش ویدئو/تصویر
قابلیت‌ها: آپلود تصویر و ویدئو، تشخیص پلاک، نمایش نتایج، دانلود خروجی
"""

import streamlit as st
import cv2
import numpy as np
import requests
import json
import base64
from datetime import datetime
from PIL import Image
import io
import time
import plotly.graph_objects as go
from pathlib import Path

# تنظیمات صفحه
st.set_page_config(
    page_title="آپلود و پردازش | سامانه تشخیص پلاک",
    page_icon="📤",
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
    .upload-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .plate-box {
        background: #1e3c72;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 18px;
        text-align: center;
    }
    .confidence-high {
        color: #00cc88;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffcc00;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# عنوان صفحه
st.markdown("""
<div class="upload-header">
    <h1>📤 آپلود و پردازش هوشمند</h1>
    <p>آپلود تصویر یا ویدئو برای تشخیص پلاک خودرو - دریافت نتایج آنی</p>
</div>
""", unsafe_allow_html=True)

# بررسی احراز هویت
if "token" not in st.session_state:
    st.error("❌ لطفاً ابتدا وارد سامانه شوید")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}
API_BASE = st.session_state.get("api_base", "http://localhost:8000")


# ==================== توابع کمکی ====================

def api_call(method: str, endpoint: str, data=None, files=None):
    """ارسال درخواست به API"""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                resp = requests.post(url, headers=headers, files=files)
            else:
                resp = requests.post(url, headers=headers, json=data)
        else:
            return None, "متد نامعتبر"
        
        if resp.status_code in [200, 201]:
            return resp.json(), None
        else:
            return None, resp.json().get("detail", "خطا در ارتباط با سرور")
    except Exception as e:
        return None, str(e)


def detect_plate_from_image(image_bytes, filename="image.jpg"):
    """تشخیص پلاک از تصویر"""
    files = {"file": (filename, image_bytes, "image/jpeg")}
    return api_call("POST", "/detect", files=files)


def process_video(video_bytes, filename="video.mp4"):
    """پردازش ویدئو و تشخیص پلاک در فریم‌ها"""
    files = {"file": (filename, video_bytes, "video/mp4")}
    return api_call("POST", "/detect/video", files=files)


def get_upload_history(limit=20):
    """دریافت تاریخچه آپلودها"""
    return api_call("GET", f"/uploads/history?limit={limit}")


def delete_upload(upload_id):
    """حذف یک آپلود"""
    return api_call("DELETE", f"/uploads/{upload_id}")


def get_plate_annotated_image(image_id):
    """دریافت تصویر با آنوتیشن پلاک"""
    return api_call("GET", f"/uploads/annotated/{image_id}")


# ==================== تب‌های اصلی ====================

tab1, tab2, tab3 = st.tabs(["📸 تصویر", "🎥 ویدئو", "📋 تاریخچه"])


# ==================== تب 1: تصویر ====================

with tab1:
    st.markdown("### 📸 تشخیص پلاک از تصویر")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # آپلود فایل
        uploaded_image = st.file_uploader(
            "انتخاب تصویر",
            type=["jpg", "jpeg", "png", "bmp"],
            help="فرمت‌های مجاز: JPG, JPEG, PNG, BMP"
        )
        
        # تنظیمات پیشرفته
        with st.expander("⚙️ تنظیمات پیشرفته"):
            enable_preprocessing = st.checkbox("پیش‌پردازش خودکار (بهبود کیفیت)", value=True)
            enable_xai = st.checkbox("نمایش نقشه حرارتی (XAI)", value=False, help="نمایش مناطق تأثیرگذار در تشخیص")
    
    with col2:
        if uploaded_image is not None:
            # نمایش تصویر آپلود شده
            image = Image.open(uploaded_image)
            st.image(image, caption="تصویر اصلی", use_container_width=True)
            
            # دکمه تشخیص
            if st.button("🔍 تشخیص پلاک", use_container_width=True):
                with st.spinner("در حال پردازش تصویر..."):
                    # تبدیل تصویر به بایت
                    img_bytes = uploaded_image.getvalue()
                    
                    # ارسال به API
                    result, error = detect_plate_from_image(img_bytes, uploaded_image.name)
                    
                    if error:
                        st.error(f"خطا در تشخیص: {error}")
                    else:
                        st.success("✅ تشخیص با موفقیت انجام شد")
                        
                        # نمایش نتایج
                        plates = result.get("plates", [])
                        
                        if plates:
                            st.markdown("### 🚗 نتایج تشخیص")
                            
                            for i, plate in enumerate(plates):
                                col_a, col_b, col_c = st.columns([2, 1, 1])
                                
                                with col_a:
                                    st.markdown(f"**پلاک {i+1}:** {plate.get('plate_text', 'نامشخص')}")
                                
                                with col_b:
                                    confidence = plate.get('confidence', 0)
                                    if confidence > 0.8:
                                        st.markdown(f"<span class='confidence-high'>اطمینان: {confidence*100:.1f}%</span>", unsafe_allow_html=True)
                                    elif confidence > 0.6:
                                        st.markdown(f"<span class='confidence-medium'>اطمینان: {confidence*100:.1f}%</span>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<span class='confidence-low'>اطمینان: {confidence*100:.1f}%</span>", unsafe_allow_html=True)
                                
                                with col_c:
                                    st.markdown(f"🕐 {plate.get('timestamp', datetime.now().strftime('%H:%M:%S'))}")
                            
                            # تصویر با آنوتیشن
                            if result.get("annotated_image"):
                                annotated_b64 = result["annotated_image"]
                                annotated_bytes = base64.b64decode(annotated_b64)
                                annotated_img = Image.open(io.BytesIO(annotated_bytes))
                                st.image(annotated_img, caption="تصویر با آنوتیشن", use_container_width=True)
                            
                            # نقشه حرارتی XAI
                            if enable_xai and result.get("heatmap"):
                                heatmap_b64 = result["heatmap"]
                                heatmap_bytes = base64.b64decode(heatmap_b64)
                                heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                                st.image(heatmap_img, caption="نقشه حرارتی (مناطق تأثیرگذار)", use_container_width=True)
                        else:
                            st.warning("⚠️ هیچ پلاکی در این تصویر تشخیص داده نشد")


# ==================== تب 2: ویدئو ====================
with tab2:
    st.markdown("### 🎥 تشخیص پلاک از ویدئو")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # آپلود ویدئو
        uploaded_video = st.file_uploader(
            "انتخاب ویدئو",
            type=["mp4", "avi", "mov", "mkv"],
            help="فرمت‌های مجاز: MP4, AVI, MOV, MKV"
        )
        
        # تنظیمات پردازش ویدئو
        with st.expander("⚙️ تنظیمات پیشرفته"):
            frame_interval = st.slider("فاصله بین فریم‌های پردازش", 1, 30, 5, help="هر چند فریم یکبار تشخیص انجام شود")
            save_annotated = st.checkbox("ذخیره ویدئوی آنوتیشن شده", value=True)
            enable_tracking = st.checkbox("فعال کردن ردیابی خودروها", value=True)
    
    with col2:
        if uploaded_video is not None:
            # نمایش پیش‌نمایش ویدئو
            st.video(uploaded_video)
            
            # دکمه پردازش
            if st.button("🎬 شروع پردازش ویدئو", use_container_width=True):
                with st.spinner("در حال پردازش ویدئو... این فرآیند ممکن است چند دقیقه طول بکشد"):
                    # ارسال ویدئو به API
                    video_bytes = uploaded_video.getvalue()
                    result, error = process_video(video_bytes, uploaded_video.name)
                    
                    if error:
                        st.error(f"خطا در پردازش: {error}")
                    else:
                        st.success("✅ پردازش ویدئو با موفقیت انجام شد")
                        
                        # نمایش آمار پردازش
                        stats = result.get("statistics", {})
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            st.metric("تعداد فریم‌ها", stats.get("total_frames", 0))
                        with col_b:
                            st.metric("تشخیص‌های انجام شده", stats.get("detections_count", 0))
                        with col_c:
                            st.metric("پلاک یکتا", stats.get("unique_plates", 0))
                        with col_d:
                            st.metric("زمان پردازش", f"{stats.get('processing_time', 0):.1f} ثانیه")
                        
                        # نمایش لیست تشخیص‌ها
                        detections = result.get("detections", [])
                        if detections:
                            st.markdown("### 📋 لیست پلاک‌های شناسایی شده")
                            
                            df_data = []
                            for det in detections:
                                df_data.append({
                                    "زمان": det.get("timestamp", "0:00"),
                                    "پلاک": det.get("plate_text", "نامشخص"),
                                    "اطمینان": f"{det.get('confidence', 0)*100:.1f}%"
                                })
                            
                            import pandas as pd
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # نمودار توزیع زمانی
                            if len(df_data) > 1:
                                timestamps = [d["زمان"] for d in df_data]
                                fig = go.Figure(data=[go.Scatter(
                                    x=list(range(len(timestamps))),
                                    y=[d["اطمینان"].replace("%", "") for d in df_data],
                                    mode='lines+markers',
                                    name='اطمینان تشخیص'
                                )])
                                fig.update_layout(
                                    title="تغییرات اطمینان تشخیص در طول ویدئو",
                                    xaxis_title="شماره تشخیص",
                                    yaxis_title="اطمینان (%)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # دانلود ویدئوی آنوتیشن شده
                        if save_annotated and result.get("annotated_video_url"):
                            st.markdown("### 📥 دانلود خروجی")
                            video_url = result["annotated_video_url"]
                            st.markdown(f"[دانلود ویدئوی آنوتیشن شده]({video_url})")


# ==================== تب 3: تاریخچه ====================

with tab3:
    st.markdown("### 📋 تاریخچه آپلود و پردازش‌ها")
    
    # دریافت تاریخچه
    history, error = get_upload_history()
    
    if error:
        st.error(f"خطا در دریافت تاریخچه: {error}")
    elif history:
        import pandas as pd
        
        # تبدیل به دیتافریم
        df = pd.DataFrame(history)
        
        # نمایش فیلترها
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox("نوع فایل", ["همه", "تصویر", "ویدئو"])
        with col2:
            filter_status = st.selectbox("وضعیت", ["همه", "موفق", "خطا"])
        with col3:
            search_text = st.text_input("جستجو در نتایج", placeholder="پلاک...")
        
        # اعمال فیلترها
        if filter_type == "تصویر":
            df = df[df["file_type"] == "image"]
        elif filter_type == "ویدئو":
            df = df[df["file_type"] == "video"]
        
        if filter_status == "موفق":
            df = df[df["status"] == "success"]
        elif filter_status == "خطا":
            df = df[df["status"] == "error"]
        
        if search_text:
            df = df[df["plates"].astype(str).str.contains(search_text)]
        
        # نمایش جدول
        st.dataframe(df, use_container_width=True)
        
        # نمایش جزئیات انتخاب شده
        if len(df) > 0:
            selected_idx = st.selectbox("انتخاب آیتم برای مشاهده جزئیات", range(len(df)))
            if selected_idx is not None:
                selected = df.iloc[selected_idx]
                
                st.markdown("### 📄 جزئیات")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**نام فایل:** {selected.get('filename', '-')}")
                    st.markdown(f"**تاریخ آپلود:** {selected.get('created_at', '-')}")
                    st.markdown(f"**وضعیت:** {'✅ موفق' if selected.get('status') == 'success' else '❌ خطا'}")
                
                with col2:
                    st.markdown(f"**نوع فایل:** {'📸 تصویر' if selected.get('file_type') == 'image' else '🎥 ویدئو'}")
                    st.markdown(f"**تعداد پلاک:** {selected.get('plate_count', 0)}")
                    st.markdown(f"**زمان پردازش:** {selected.get('processing_time', '-')}")
                
                if selected.get("plates"):
                    st.markdown("**پلاک‌های شناسایی شده:**")
                    for plate in selected["plates"]:
                        st.markdown(f"- {plate}")
                
                # دکمه حذف
                if st.button("🗑️ حذف این آیتم", use_container_width=True):
                    result, error = delete_upload(selected["id"])
                    if error:
                        st.error(f"خطا: {error}")
                    else:
                        st.success("آیتم با موفقیت حذف شد")
                        st.rerun()
    else:
        st.info("هیچ سابقه‌ای یافت نشد")


# ==================== راهنمای استفاده ====================

with st.expander("📖 راهنمای استفاده"):
    st.markdown("""
    ### نحوه استفاده از صفحه آپلود
    
    برای تشخیص پلاک از تصویر:
    1. در تب "تصویر"، فایل تصویر خود را آپلود کنید
    2. در صورت نیاز، تنظیمات پیشرفته را تغییر دهید
    3. روی دکمه "تشخیص پلاک" کلیک کنید
    4. نتایج شامل پلاک‌های شناسایی شده و تصویر آنوتیشن شده نمایش داده می‌شود
    
    برای تشخیص پلاک از ویدئو:
    1. در تب "ویدئو"، فایل ویدئوی خود را آپلود کنید
    2. فاصله فریم‌ها و سایر تنظیمات را مشخص کنید
    3. روی دکمه "شروع پردازش ویدئو" کلیک کنید
    4. پس از اتمام پردازش، آمار و نتایج نمایش داده می‌شود
    
    مشاهده تاریخچه:
    - در تب "تاریخچه" می‌توانید تمام آپلودهای قبلی را مشاهده کنید
    - می‌توانید بر اساس نوع فایل، وضعیت و متن جستجو فیلتر کنید
    - برای مشاهده جزئیات، روی آیتم مورد نظر کلیک کنید
    """)

# ==================== فوتر ====================

st.markdown("---")
st.caption(f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")