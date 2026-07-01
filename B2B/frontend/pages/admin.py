"""
admin.py - پنل مدیریت ادمین (فرانت‌اند استریملیت)
قابلیت‌ها: مدیریت سازمان‌ها، کاربران، دوربین‌ها، مشاهده آمار و سلامت سیستم
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# تنظیمات صفحه
st.set_page_config(
    page_title="پنل مدیریت | سامانه تشخیص پلاک",
    page_icon="👑",
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
    .css-1v0mbdj, .css-1rs6os {
        direction: rtl;
    }
    .main-header {
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
</style>
""", unsafe_allow_html=True)

# عنوان صفحه
st.markdown("""
<div class="main-header">
    <h1>👑 پنل مدیریت سازمانی</h1>
    <p>مدیریت سازمان‌ها، کاربران، دوربین‌ها و نظارت بر عملکرد سیستم</p>
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
        elif method == "PUT":
            resp = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers)
        else:
            return None, "متد نامعتبر"
        
        if resp.status_code in [200, 201]:
            return resp.json(), None
        else:
            return None, resp.json().get("detail", "خطا در ارتباط با سرور")
    except Exception as e:
        return None, str(e)


# ==================== منوی کناری ====================

st.sidebar.markdown("## 📋 منوی مدیریت")
menu = st.sidebar.radio(
    "انتخاب بخش",
    ["🏢 سازمان‌ها", "👥 کاربران", "📷 دوربین‌ها", "📊 آمار و داشبورد", "💳 لایسنس", "🩺 سلامت سیستم"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**وضعیت:** {'🟢 متصل' if st.session_state.token else '🔴 قطع'}")
if st.sidebar.button("🚪 خروج از پنل"):
    st.session_state.token = None
    st.rerun()


# ==================== 1. مدیریت سازمان‌ها ====================

if menu == "🏢 سازمان‌ها":
    st.header("🏢 مدیریت سازمان‌ها")
    
    tab1, tab2, tab3 = st.tabs(["📋 لیست سازمان‌ها", "➕ افزودن سازمان", "✏️ ویرایش سازمان"])
    
    # تب 1: لیست سازمان‌ها
    with tab1:
        orgs, error = api_call("GET", "/admin/organizations")
        if error:
            st.error(f"خطا: {error}")
        else:
            if orgs:
                df = pd.DataFrame(orgs)
                st.dataframe(df, use_container_width=True)
                
                # نمایش جزئیات سازمان انتخاب شده
                selected_org = st.selectbox("انتخاب سازمان برای مشاهده جزئیات", [o["name"] for o in orgs])
                if selected_org:
                    org_data = next(o for o in orgs if o["name"] == selected_org)
                    st.json(org_data)
            else:
                st.info("هیچ سازمانی یافت نشد")
    
    # تب 2: افزودن سازمان
    with tab2:
        with st.form("add_org_form"):
            org_name = st.text_input("نام سازمان")
            org_tier = st.selectbox("نوع اشتراک", ["standard", "pro", "enterprise"])
            org_max_cameras = st.number_input("حداکثر تعداد دوربین", min_value=1, max_value=100, value=5)
            org_quota = st.number_input("سقف تشخیص روزانه", min_value=100, max_value=1000000, value=5000)
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("✅ ایجاد سازمان")
            with col2:
                cancelled = st.form_submit_button("❌ انصراف")
            
            if submitted:
                data = {
                    "name": org_name,
                    "tier": org_tier,
                    "max_cameras": org_max_cameras,
                    "quota_limit": org_quota
                }
                result, error = api_call("POST", "/admin/organizations", data)
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.success(f"سازمان {org_name} با موفقیت ایجاد شد")
                    st.rerun()
    
    # تب 3: ویرایش سازمان
    with tab3:
        orgs, error = api_call("GET", "/admin/organizations")
        if orgs:
            org_names = [o["name"] for o in orgs]
            selected_org = st.selectbox("انتخاب سازمان برای ویرایش", org_names)
            if selected_org:
                org_data = next(o for o in orgs if o["name"] == selected_org)
                with st.form("edit_org_form"):
                    new_name = st.text_input("نام جدید", value=org_data["name"])
                    new_tier = st.selectbox("نوع اشتراک جدید", ["standard", "pro", "enterprise"], index=["standard", "pro", "enterprise"].index(org_data["tier"]))
                    new_max_cameras = st.number_input("حداکثر دوربین جدید", value=org_data["max_cameras"])
                    is_active = st.checkbox("فعال", value=org_data.get("is_active", True))
                    
                    if st.form_submit_button("💾 ذخیره تغییرات"):
                        data = {
                            "name": new_name,
                            "tier": new_tier,
                            "max_cameras": new_max_cameras,
                            "is_active": is_active
                        }
                        result, error = api_call("PUT", f"/admin/organizations/{org_data['id']}", data)
                        if error:
                            st.error(f"خطا: {error}")
                        else:
                            st.success("تغییرات ذخیره شد")
                            st.rerun()


# ==================== 2. مدیریت کاربران ====================

elif menu == "👥 کاربران":
    st.header("👥 مدیریت کاربران")
    
    # انتخاب سازمان
    orgs, error = api_call("GET", "/admin/organizations")
    if error or not orgs:
        st.error("امکان بارگذاری لیست سازمان‌ها")
    else:
        org_names = [o["name"] for o in orgs]
        selected_org = st.selectbox("انتخاب سازمان", org_names)
        org_id = next(o["id"] for o in orgs if o["name"] == selected_org)
        
        tab1, tab2 = st.tabs(["📋 لیست کاربران", "➕ افزودن کاربر"])
        
        with tab1:
            users, error = api_call("GET", f"/admin/organizations/{org_id}/users")
            if error:
                st.error(f"خطا: {error}")
            elif users:
                df = pd.DataFrame(users)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("هیچ کاربری در این سازمان یافت نشد")
        
        with tab2:
            with st.form("add_user_form"):
                user_email = st.text_input("ایمیل")
                user_password = st.text_input("رمز عبور", type="password")
                user_fullname = st.text_input("نام کامل")
                user_role = st.selectbox("نقش", ["admin", "operator", "viewer"])
                
                if st.form_submit_button("➕ افزودن کاربر"):
                    data = {
                        "email": user_email,
                        "password": user_password,
                        "full_name": user_fullname,
                        "role": user_role
                    }
                    result, error = api_call("POST", f"/admin/organizations/{org_id}/users", data)
                    if error:
                        st.error(f"خطا: {error}")
                    else:
                        st.success("کاربر با موفقیت افزوده شد")
                        st.rerun()


# ==================== 3. مدیریت دوربین‌ها ====================

elif menu == "📷 دوربین‌ها":
    st.header("📷 مدیریت دوربین‌ها")
    
    orgs, error = api_call("GET", "/admin/organizations")
    if error or not orgs:
        st.error("امکان بارگذاری لیست سازمان‌ها")
    else:
        org_names = [o["name"] for o in orgs]
        selected_org = st.selectbox("انتخاب سازمان", org_names)
        org_id = next(o["id"] for o in orgs if o["name"] == selected_org)
        
        tab1, tab2 = st.tabs(["📋 لیست دوربین‌ها", "➕ افزودن دوربین"])
        
        with tab1:
            cameras, error = api_call("GET", f"/admin/organizations/{org_id}/cameras")
            if error:
                st.error(f"خطا: {error}")
            elif cameras:
                df = pd.DataFrame(cameras)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("هیچ دوربینی در این سازمان یافت نشد")
        
        with tab2:
            with st.form("add_camera_form"):
                cam_name = st.text_input("نام دوربین")
                cam_stream_url = st.text_input("آدرس استریم (RTSP / HTTP)")
                cam_stream_type = st.selectbox("نوع استریم", ["rtsp", "http", "usb", "v4l2"])
                cam_location = st.text_input("موقعیت مکانی (اختیاری)")
                
                if st.form_submit_button("➕ افزودن دوربین"):
                    data = {
                        "name": cam_name,
                        "stream_url": cam_stream_url,
                        "stream_type": cam_stream_type,
                        "location": cam_location
                    }
                    result, error = api_call("POST", f"/admin/organizations/{org_id}/cameras", data)
                    if error:
                        st.error(f"خطا: {error}")
                    else:
                        st.success("دوربین با موفقیت افزوده شد")
                        st.rerun()


# ==================== 4. آمار و داشبورد ====================

elif menu == "📊 آمار و داشبورد":
    st.header("📊 آمار و داشبورد")
    
    stats, error = api_call("GET", "/admin/dashboard/stats")
    if error:
        st.error(f"خطا: {error}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_organizations', 0)}</div>
                <div class="stat-label">سازمان‌ها</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_users', 0)}</div>
                <div class="stat-label">کاربران</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_cameras', 0)}</div>
                <div class="stat-label">دوربین‌ها</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats.get('total_detections_today', 0)}</div>
                <div class="stat-label">تشخیص امروز</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📈 روند تشخیص پلاک")
        
        # دریافت داده‌های روند (در صورت وجود API)
        detections, _ = api_call("GET", "/admin/dashboard/recent-detections")
        if detections:
            df = pd.DataFrame(detections)
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"])
                df["date"] = df["created_at"].dt.date
                daily = df.groupby("date").size().reset_index(name="count")
                fig = px.line(daily, x="date", y="count", title="تعداد تشخیص در روزهای اخیر")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("داده‌ای برای نمایش وجود ندارد")
    
    st.markdown("---")
    st.caption(f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==================== 5. مدیریت لایسنس ====================

elif menu == "💳 لایسنس":
    st.header("💳 مدیریت لایسنس سازمان‌ها")
    
    orgs, error = api_call("GET", "/admin/organizations")
    if error or not orgs:
        st.error("امکان بارگذاری لیست سازمان‌ها")
    else:
        org_names = [o["name"] for o in orgs]
        selected_org = st.selectbox("انتخاب سازمان", org_names)
        org_id = next(o["id"] for o in orgs if o["name"] == selected_org)
        
        license_info, error = api_call("GET", f"/admin/license/{org_id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 اطلاعات لایسنس فعلی")
            if license_info and not error:
                st.metric("حداکثر دوربین مجاز", license_info.get("max_cameras", "?"))
                st.metric("تاریخ انقضا", license_info.get("expires_at", "?"))
                st.metric("روزهای باقیمانده", license_info.get("days_remaining", "?"))
                if license_info.get("is_valid"):
                    st.success("✅ لایسنس معتبر است")
                else:
                    st.error("❌ لایسنس منقضی شده است")
        
        with col2:
            st.subheader("🔄 تمدید لایسنس")
            new_max_cameras = st.number_input("حداکثر دوربین جدید", min_value=1, max_value=100, value=10)
            new_days = st.number_input("مدت اعتبار (روز)", min_value=30, max_value=3650, value=365)
            
            if st.button("🔄 تمدید لایسنس"):
                result, error = api_call("POST", f"/admin/license/generate", {
                    "org_id": org_id,
                    "max_cameras": new_max_cameras,
                    "expiry_days": new_days
                })
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.success("لایسنس با موفقیت تمدید شد")
                    st.code(result.get("license_token", ""))
                    st.rerun()


# ==================== 6. سلامت سیستم ====================

elif menu == "🩺 سلامت سیستم":
    st.header("🩺 سلامت سیستم")
    
    health, error = api_call("GET", "/admin/health")
    
    if error:
        st.error(f"خطا در ارتباط با سرور: {error}")
    else:
        col1, col2, col3 = st.columns(3)
        
        status_color = "🟢" if health.get("status") == "healthy" else "🟡" if health.get("status") == "degraded" else "🔴"
        
        with col1:
            st.metric("وضعیت کلی", f"{status_color} {health.get('status', 'unknown')}")
        with col2:
            st.metric("پایگاه داده", "🟢 سالم" if health.get("database") == "healthy" else "🔴 مشکل")
        with col3:
            st.metric("Redis", "🟢 سالم" if health.get("redis") == "healthy" else "🔴 مشکل")
        
        st.markdown("---")
        
        if health.get("gpu_available"):
            st.success(f"✅ GPU فعال است - حافظه استفاده شده: {health.get('gpu_memory_used_gb', 0):.2f} GB")
        else:
            st.info("ℹ️ GPU در این سیستم موجود نیست - پردازش روی CPU انجام می‌شود")
        
        # نمایش نمودار CPU و RAM
        col1, col2 = st.columns(2)
        with col1:
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health.get("cpu_percent", 0),
                title={"text": "استفاده CPU (%)"},
                domain={"x": [0, 1], "y": [0, 1]}
            ))
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            fig_ram = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health.get("memory_percent", 0),
                title={"text": "استفاده RAM (%)"},
                domain={"x": [0, 1], "y": [0, 1]}
            ))
            st.plotly_chart(fig_ram, use_container_width=True)
        
        st.caption(f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        