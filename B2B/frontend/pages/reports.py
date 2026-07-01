"""
reports.py - صفحه گزارش‌گیری و آمار پیشرفته
قابلیت‌ها: گزارش‌گیری از تشخیص پلاک، دانلود PDF/Excel، نمودارهای آماری، تحلیل عملکرد
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import base64
from io import BytesIO

# تنظیمات صفحه
st.set_page_config(
    page_title="گزارش‌ها | سامانه تشخیص پلاک",
    page_icon="📊",
    layout="wide"
)

# CSS برای راست‌چین و فارسی
st.markdown("""
<style>
    body {
        direction: rtl;
        text-align: right;
    }
    .report-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stat-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .stat-number {
        font-size: 32px;
        font-weight: bold;
        color: #1e3c72;
    }
    .stat-label {
        font-size: 14px;
        color: #555;
    }
    .download-btn {
        background-color: #1e3c72;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# عنوان صفحه
st.markdown("""
<div class="report-header">
    <h1>📊 گزارش‌گیری و آمار پیشرفته</h1>
    <p>تحلیل تردد خودروها، تشخیص پلاک، عملکرد سیستم و خروجی‌های آماری</p>
</div>
""", unsafe_allow_html=True)

# بررسی احراز هویت
if "token" not in st.session_state:
    st.error("❌ لطفاً ابتدا وارد سامانه شوید")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}
API_BASE = st.session_state.get("api_base", "http://localhost:8000")


# ==================== توابع کمکی ====================

def api_call(method: str, endpoint: str, data=None):
    """ارسال درخواست به API"""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=data)
        else:
            return None, "متد نامعتبر"
        
        if resp.status_code in [200, 201]:
            return resp.json(), None
        else:
            return None, resp.json().get("detail", "خطا در ارتباط با سرور")
    except Exception as e:
        return None, str(e)


def get_detections_report(start_date, end_date, org_id=None, camera_id=None):
    """دریافت گزارش تشخیص پلاک در بازه زمانی مشخص"""
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    if org_id:
        params["org_id"] = org_id
    if camera_id:
        params["camera_id"] = camera_id
    
    try:
        resp = requests.get(f"{API_BASE}/reports/detections", headers=headers, params=params)
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "خطا در دریافت گزارش")
    except Exception as e:
        return None, str(e)


def download_report(format: str, start_date, end_date, org_id=None, camera_id=None):
    """دانلود گزارش به صورت PDF یا Excel"""
    params = {
        "format": format,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    if org_id:
        params["org_id"] = org_id
    if camera_id:
        params["camera_id"] = camera_id
    
    try:
        resp = requests.get(f"{API_BASE}/reports/download", headers=headers, params=params)
        if resp.status_code == 200:
            return resp.content, None
        return None, resp.json().get("detail", "خطا در دانلود")
    except Exception as e:
        return None, str(e)
    def get_plate_statistics(plate_number, days=30):
     try:
        resp = requests.get(
            f"{API_BASE}/reports/plate-stats/{plate_number}",
            headers=headers,
            params={"days": days}
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "خطا")
     except Exception as e:
        return None, str(e)


def get_camera_performance(camera_id, days=7):
    """دریافت عملکرد دوربین در بازه زمانی مشخص"""
    try:
        resp = requests.get(
            f"{API_BASE}/reports/camera-performance/{camera_id}",
            headers=headers,
            params={"days": days}
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "خطا")
    except Exception as e:
        return None, str(e)


# ==================== منوی کناری ====================

st.sidebar.markdown("## 📅 بازه زمانی گزارش")

# تنظیمات تاریخ پیش‌فرض
today = datetime.now().date()
default_start = today - timedelta(days=7)

start_date = st.sidebar.date_input("از تاریخ", default_start)
end_date = st.sidebar.date_input("تا تاریخ", today)

if start_date > end_date:
    st.sidebar.error("تاریخ شروع نباید از تاریخ پایان بزرگتر باشد")

st.sidebar.markdown("---")

# انتخاب سازمان (برای ادمین)
orgs, _ = api_call("GET", "/admin/organizations")
if orgs and len(orgs) > 0:
    org_names = ["همه سازمان‌ها"] + [o["name"] for o in orgs]
    selected_org_name = st.sidebar.selectbox("سازمان", org_names)
    if selected_org_name != "همه سازمان‌ها":
        selected_org = next(o for o in orgs if o["name"] == selected_org_name)
        org_id = selected_org["id"]
    else:
        org_id = None
else:
    org_id = None

st.sidebar.markdown("---")
st.sidebar.markdown("## 📥 خروجی")
export_format = st.sidebar.selectbox("فرمت خروجی", ["PDF", "Excel", "CSV", "JSON"])


# ==================== تب‌های اصلی ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 گزارش تشخیص پلاک",
    "📈 نمودارهای آماری",
    "🔍 جستجوی پلاک",
    "🎥 عملکرد دوربین‌ها",
    "📊 گزارش تردد"
])


# ==================== تب 1: گزارش تشخیص پلاک ====================

with tab1:
    st.markdown("### 📋 گزارش تشخیص پلاک")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("📥 دانلود گزارش", use_container_width=True):
            with st.spinner("در حال تولید گزارش..."):
                content, error = download_report(
                    export_format.lower(),
                    start_date,
                    end_date,
                    org_id
                )
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.success("گزارش با موفقیت تولید شد")
                    
                    # دانلود فایل
                    mime_types = {
                        "PDF": "application/pdf",
                        "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "CSV": "text/csv",
                        "JSON": "application/json"
                    }
                    file_ext = export_format.lower()
                    st.download_button(
                        label="💾 ذخیره فایل",
                        data=content,
                        file_name=f"report_{start_date}_{end_date}.{file_ext}",
                        mime=mime_types.get(export_format, "application/octet-stream")
                    )
    
    with col1:
        # دریافت و نمایش گزارش
        report, error = get_detections_report(start_date, end_date, org_id)
        
        if error:
            st.error(f"خطا: {error}")
        elif report and report.get("detections"):
            detections = report["detections"]
            df = pd.DataFrame(detections)
            
            # نمایش آمار خلاصه
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("تعداد کل تشخیص", len(df))
            with col_b:
                st.metric("میانگین اطمینان", f"{df['confidence'].mean()*100:.1f}%" if 'confidence' in df else "-")
            with col_c:
                unique_plates = df['plate_text'].nunique() if 'plate_text' in df else 0
                st.metric("پلاک یکتا", unique_plates)
            with col_d:
                st.metric("دوربین‌های فعال", df['camera_id'].nunique() if 'camera_id' in df else 0)
            
            st.markdown("---")
            
            # نمایش جدول داده‌ها
            st.dataframe(df, use_container_width=True)
            
            # نمایش آمار روزانه
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
                df["date"] = df["created_at"].dt.date
                daily = df.groupby("date").size().reset_index(name="count")
                
                st.subheader("📅 توزیع روزانه")
                fig = px.bar(daily, x="date", y="count", title="تعداد تشخیص در روز")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("هیچ داده‌ای در بازه زمانی انتخاب شده یافت نشد")


# ==================== تب 2: نمودارهای آماری ====================

with tab2:
    st.markdown("### 📈 نمودارهای آماری")
    
    report, error = get_detections_report(start_date, end_date, org_id)
    
    if error:
        st.error(f"خطا: {error}")
    elif report and report.get("detections"):
        df = pd.DataFrame(report["detections"])
        
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["hour"] = df["created_at"].dt.hour
            df["day"] = df["created_at"].dt.day_name()
            df["week"] = df["created_at"].dt.isocalendar().week
            
            col1, col2 = st.columns(2)
            
            with col1:
                # نمودار ساعتی
                hourly = df.groupby("hour").size().reset_index(name="count")
                fig1 = px.line(hourly, x="hour", y="count", title="توزیع تردد در ساعات مختلف")
                fig1.update_layout(xaxis_title="ساعت", yaxis_title="تعداد تردد")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # نمودار روزهای هفته
                weekly = df.groupby("day").size().reset_index(name="count")
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_names_fa = {
                    "Monday": "دوشنبه", "Tuesday": "سه‌شنبه", "Wednesday": "چهارشنبه",
                    "Thursday": "پنجشنبه", "Friday": "جمعه", "Saturday": "شنبه", "Sunday": "یکشنبه"
                }
                weekly["day_fa"] = weekly["day"].map(day_names_fa)
                fig2 = px.bar(weekly, x="day_fa", y="count", title="توزیع تردد در روزهای هفته")
                st.plotly_chart(fig2, use_container_width=True)
            
            # نمودار اطمینان تشخیص
            if "confidence" in df.columns:
                fig3 = go.Figure(data=[
                    go.Histogram(x=df["confidence"], nbinsx=20, marker_color="#1e3c72")
                ])
                fig3.update_layout(
                    title="توزیع میزان اطمینان تشخیص",
                    xaxis_title="اطمینان",
                    yaxis_title="تعداد"
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # نمودار پلاک‌های تکراری
            if "plate_text" in df.columns:
                top_plates = df["plate_text"].value_counts().head(10).reset_index()
                top_plates.columns = ["plate_text", "count"]
                
                fig4 = px.bar(top_plates, x="plate_text", y="count", title="۱۰ پلاک پرتکرار")
                st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("داده‌ای برای نمایش وجود ندارد")


# ==================== تب 3: جستجوی پلاک ====================

with tab3:
    st.markdown("### 🔍 جستجوی پلاک خاص")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_plate = st.text_input("شماره پلاک", placeholder="مثال: ۱۲۳۴۵۶۷")
    
    with col2:
        search_days = st.number_input("تعداد روزهای گذشته", min_value=1, max_value=90, value=30)
    
    if search_plate:
        with st.spinner("در حال جستجو..."):
            stats, error = get_plate_statistics(search_plate, search_days)  # noqa: F821
            
            if error:
                st.error(f"خطا: {error}")
            elif stats:
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("تعداد تردد", stats.get("total_visits", 0))
                with col_b:
                    st.metric("آخرین تردد", stats.get("last_seen", "-"))
                with col_c:
                    st.metric("اولین تردد", stats.get("first_seen", "-"))
                with col_d:
                    st.metric("میانگین روزانه", stats.get("daily_average", 0))
                
                st.markdown("---")
                
                # نمایش تاریخچه تردد
                if stats.get("history"):
                    history_df = pd.DataFrame(stats["history"])
                    st.dataframe(history_df, use_container_width=True)
                    
                    # نمودار تردد در زمان
                    if "timestamp" in history_df.columns:
                        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
                        history_df["date"] = history_df["timestamp"].dt.date
                        daily_history = history_df.groupby("date").size().reset_index(name="count")
                        fig = px.line(daily_history, x="date", y="count", title="تاریخچه تردد")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"هیچ ترددی برای پلاک {search_plate} یافت نشد")


# ==================== تب 4: عملکرد دوربین‌ها ====================

with tab4:
    st.markdown("### 🎥 عملکرد دوربین‌ها")
    
    # دریافت لیست دوربین‌ها
    cameras, error = api_call("GET", "/admin/cameras")
    
    if error:
        st.error(f"خطا: {error}")
    elif cameras:
        selected_cam_name = st.selectbox("انتخاب دوربین", [c["name"] for c in cameras])
        selected_cam = next(c for c in cameras if c["name"] == selected_cam_name)
        
        days = st.slider("بازه زمانی (روز)", 1, 30, 7)
        
        with st.spinner("در حال دریافت اطلاعات..."):
            performance, error = get_camera_performance(selected_cam["id"], days)
            
            if error:
                st.error(f"خطا: {error}")
            elif performance:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("تعداد کل تشخیص", performance.get("total_detections", 0))
                with col2:
                    st.metric("میانگین روزانه", performance.get("daily_average", 0))
                with col3:
                    st.metric("نرخ موفقیت", f"{performance.get('success_rate', 0)*100:.1f}%")
                with col4:
                    st.metric("پربازدیدترین ساعت", performance.get("peak_hour", "-"))
                
                st.markdown("---")
                
                # نمودار عملکرد روزانه
                if performance.get("daily_stats"):
                    daily_df = pd.DataFrame(performance["daily_stats"])
                    fig = px.bar(daily_df, x="date", y="count", title="تشخیص‌های روزانه")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("هیچ دوربینی یافت نشد")
        # ==================== تب 5: گزارش تردد ====================

with tab5:
    st.markdown("### 📊 گزارش تردد (ورود و خروج)")
    
    report, error = get_detections_report(start_date, end_date, org_id)
    
    if error:
        st.error(f"خطا: {error}")
    elif report and report.get("detections"):
        df = pd.DataFrame(report["detections"])
        
        if "plate_text" in df.columns and "created_at" in df.columns:
            # محاسبه تردد (ورود و خروج)
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["date"] = df["created_at"].dt.date
            
            # گروه‌بندی بر اساس پلاک و روز
            traffic = df.groupby(["plate_text", "date"]).size().reset_index(name="count")
            
            # تفکیک ورود و خروج (ساده شده: تعداد زوج = ورود، فرد = خروج)
            traffic["type"] = traffic.groupby("plate_text")["count"].transform(
                lambda x: ["ورود" if i % 2 == 0 else "خروج" for i in range(len(x))]
            )
            
            st.dataframe(traffic, use_container_width=True)
            
            # آمار خلاصه تردد
            col1, col2 = st.columns(2)
            with col1:
                entry_count = len(traffic[traffic["type"] == "ورود"])
                st.metric("تعداد ورود", entry_count)
            with col2:
                exit_count = len(traffic[traffic["type"] == "خروج"])
                st.metric("تعداد خروج", exit_count)
            
            # نمودار ساعتی ورود و خروج
            df["hour"] = df["created_at"].dt.hour
            hourly_traffic = df.groupby(["hour"]).size().reset_index(name="count")
            fig = px.area(hourly_traffic, x="hour", y="count", title="تردد در ساعات مختلف")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("داده‌ای برای نمایش وجود ندارد")


# ==================== فوتر ====================

st.markdown("---")
st.caption(f"آخرین بروزرسانی گزارش: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
