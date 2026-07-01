"""
app.py - نقطه ورود اصلی سامانه تشخیص پلاک فارسی
این فایل به عنوان صفحه اصلی و مدیریت احراز هویت عمل می‌کند
"""

import streamlit as st
import requests
import json
from datetime import datetime
from PIL import Image
import io
import base64

# تنظیمات صفحه - باید اولین دستور باشد
st.set_page_config(
    page_title="سامانه هوشمند تشخیص پلاک خودرو | ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== استایل‌های CSS ====================
def load_css():
    """بارگذاری استایل‌های سفارشی"""
    try:
        with open("static/style.css", "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # استایل‌های پایه در صورت نبودن فایل
        st.markdown("""
        <style>
            body { direction: rtl; text-align: right; font-family: 'Vazirmatn', 'Tahoma', sans-serif; }
            .stButton button { width: 100%; }
            .main-header { background: linear-gradient(90deg, #1e3c72, #2a5298); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .stat-card { background: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; margin: 10px; }
            .stat-number { font-size: 32px; font-weight: bold; color: #1e3c72; }
            .plate-result { background: #1e3c72; color: white; padding: 10px; border-radius: 8px; margin: 5px 0; font-family: monospace; font-size: 18px; text-align: center; }
        </style>
        """, unsafe_allow_html=True)

# بارگذاری استایل
load_css()


# ==================== توابع احراز هویت ====================

def init_session_state():
    """مقداردهی اولیه session_state"""
    if "token" not in st.session_state:
        st.session_state.token = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "user_fullname" not in st.session_state:
        st.session_state.user_fullname = None
    if "org_id" not in st.session_state:
        st.session_state.org_id = None
    if "api_base" not in st.session_state:
        st.session_state.api_base = "http://localhost:8000"
    if "last_login" not in st.session_state:
        st.session_state.last_login = None


def login(email: str, password: str):
    """ورود به سامانه"""
    try:
        response = requests.post(
            f"{st.session_state.api_base}/auth/login",
            json={"email": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data.get("access_token")
            st.session_state.user_email = email
            st.session_state.user_role = data.get("role", "operator")
            st.session_state.user_fullname = data.get("full_name", email)
            st.session_state.org_id = data.get("org_id")
            st.session_state.last_login = datetime.now()
            return True, None
        else:
            return False, response.json().get("detail", "خطا در ورود")
    except Exception as e:
        return False, str(e)


def logout():
    """خروج از سامانه"""
    if st.session_state.token:
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            requests.post(f"{st.session_state.api_base}/auth/logout", headers=headers)
        except:
            pass
    # پاک کردن session_state
    for key in ["token", "user_email", "user_role", "user_fullname", "org_id", "last_login"]:
        if key in st.session_state:
            st.session_state[key] = None
    st.rerun()


def register(email: str, password: str, full_name: str, org_name: str):
    """ثبت نام کاربر جدید"""
    try:
        response = requests.post(
            f"{st.session_state.api_base}/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name,
                "org_name": org_name
            }
        )
        if response.status_code == 201:
            data = response.json()
            st.session_state.token = data.get("access_token")
            st.session_state.user_email = email
            st.session_state.user_role = "admin"
            st.session_state.user_fullname = full_name
            st.session_state.org_id = data.get("org_id")
            st.session_state.last_login = datetime.now()
            return True, None
        else:
            return False, response.json().get("detail", "خطا در ثبت نام")
    except Exception as e:
        return False, str(e)


# ==================== هدر صفحه ====================

def show_header():
    """نمایش هدر صفحه"""
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🚘 سامانه هوشمند تشخیص پلاک خودروهای ایران</h1>
        <p style="color: #e2e8f0; margin: 10px 0 0 0;">پیشرفته‌ترین سیستم تشخیص پلاک با قابلیت‌های هوش مصنوعی</p>
    </div>
    """, unsafe_allow_html=True)


def show_user_info():
    """نمایش اطلاعات کاربر در کناری"""
    if st.session_state.user_email:
        st.sidebar.markdown(f"### 👤 {st.session_state.user_fullname}")
        st.sidebar.markdown(f"📧 {st.session_state.user_email}")
        st.sidebar.markdown(f"🎭 نقش: {st.session_state.user_role}")
        
        # نمایش لوگوی سامانه
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: 48px;">🚗</div>
            <div style="font-weight: bold; color: #1e3c72;">ANPR ایران</div>
            <div style="font-size: 12px; color: #666;">نسخه 3.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 خروج از سامانه", use_container_width=True):
            logout()


# ==================== صفحه اصلی ====================

def show_home_page():
    """نمایش صفحه اصلی با آمار و اطلاعات"""
    st.markdown("## 🏠 داشبورد اصلی")
    
    # آمار سامانه (در صورت وجود API)
    if st.session_state.token:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        try:
            stats_response = requests.get(
                f"{st.session_state.api_base}/admin/dashboard/stats",
                headers=headers
            )
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{stats.get('total_cameras', 0)}</div>
                        <div class="stat-label">دوربین‌های فعال</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{stats.get('total_detections_today', 0)}</div>
                        <div class="stat-label">تشخیص امروز</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{stats.get('total_users', 0)}</div>
                        <div class="stat-label">کاربران</div>
                    </div>
                    """, unsafe_allow_html=True)
                    with col4:
                     st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{stats.get('total_organizations', 0)}</div>
                        <div class="stat-label">سازمان‌ها</div>
                    </div>
                    """, unsafe_allow_html=True)
        except:
            pass
    
    st.markdown("---")
    
    # اطلاعات سامانه
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✨ قابلیت‌های سامانه")
        st.markdown("""
        - 🔍 تشخیص خودکار پلاک خودروهای ایرانی با دقت بالا
        - 📸 پشتیبانی از تصاویر و ویدئو با فرمت‌های مختلف
        - 🌙 عملکرد در نور کم و شرایط نامساعد جوی
        - 🚨 تشخیص ناهنجاری و هشدار لحظه‌ای
        - 📊 گزارش‌گیری پیشرفته و خروجی PDF/Excel
        - 🔔 اعلان‌های خودکار از طریق ایمیل و تلگرام
        - 🎯 ردیابی چند شیء همزمان
        - 🔒 امنیت بالا و احراز هویت دو مرحله‌ای
        """)
    
    with col2:
        st.markdown("### 🚀 شروع سریع")
        st.markdown("""
        1. برای تشخیص پلاک به بخش آپلود و پردازش بروید
        2. برای مشاهده دوربین‌ها به بخش پایش زنده مراجعه کنید
        3. برای دریافت گزارش به بخش گزارش‌ها بروید
        4. برای تنظیمات سیستم به بخش تنظیمات مراجعه کنید
        
        ---
        
        💡 نکته: در صورت نیاز به راهنمایی، به بخش تنظیمات و راهنما مراجعه کنید.
        """)
    
    st.markdown("---")
    
    # نمایش آخرین تشخیص‌ها
    if st.session_state.token:
        st.markdown("### 📋 آخرین تشخیص‌ها")
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            detections_response = requests.get(
                f"{st.session_state.api_base}/admin/dashboard/recent-detections",
                headers=headers
            )
            if detections_response.status_code == 200:
                detections = detections_response.json()
                if detections:
                    for det in detections[:5]:
                        st.markdown(f"""
                        <div class="plate-result">
                            🚗 پلاک: {det.get('plate_text', 'نامشخص')} | 
                            اطمینان: {det.get('confidence', 0)*100:.1f}% | 
                            زمان: {det.get('created_at', 'نامشخص')}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("هیچ تشخیصی ثبت نشده است")
        except:
            st.info("در حال حاضر قادر به دریافت آخرین تشخیص‌ها نیستیم")


# ==================== صفحه ورود ====================

def show_login_page():
    """نمایش صفحه ورود"""
    st.markdown("## 🔐 ورود به سامانه")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("login_form"):
            email = st.text_input("ایمیل", placeholder="example@domain.com")
            password = st.text_input("رمز عبور", type="password", placeholder="••••••••")
            
            col_a, col_b = st.columns(2)
            with col_a:
                submit = st.form_submit_button("ورود", use_container_width=True)
            with col_b:
                show_register = st.form_submit_button("ثبت نام", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("لطفاً ایمیل و رمز عبور را وارد کنید")
                else:
                    success, error = login(email, password)
                    if success:
                        st.success("ورود موفقیت آمیز بود")
                        st.rerun()
                    else:
                        st.error(f"خطا: {error}")
            
            if show_register:
                st.session_state.show_register = True
                st.rerun()
    
    with col2:
        st.markdown("""
        ### 📢 اطلاعیه
        
        برای دسترسی به سامانه، لطفاً با ایمیل و رمز عبور خود وارد شوید.
        
        در صورت نداشتن حساب کاربری، روی دکمه ثبت نام کلیک کنید.
        
        ---
        
        ### 🔧 نیاز به کمک؟
        
        در صورت بروز مشکل، با پشتیبانی تماس بگیرید:
        - 📞 تلفن: ۰۲۱-۱۲۳۴۵۶۷۸
        - 📧 ایمیل: support@anpr.ir
        """)


# ==================== صفحه ثبت نام ====================

def show_register_page():
    """نمایش صفحه ثبت نام"""
    st.markdown("## 📝 ثبت نام در سامانه")
    
    with st.form("register_form"):
        email = st.text_input("ایمیل", placeholder="example@domain.com")
        full_name = st.text_input("نام کامل", placeholder="علی محمدی")
        password = st.text_input("رمز عبور", type="password", placeholder="••••••••")
        confirm_password = st.text_input("تکرار رمز عبور", type="password")
        org_name = st.text_input("نام سازمان", placeholder="نام شرکت یا سازمان شما")
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("ثبت نام", use_container_width=True)
        with col2:
            back = st.form_submit_button("بازگشت به ورود", use_container_width=True)
        
        if submit:
            if not all([email, full_name, password, org_name]):
                st.error("لطفاً تمام فیلدها را پر کنید")
            elif password != confirm_password:
                st.error("رمز عبور و تکرار آن مطابقت ندارند")
            elif len(password) < 6:
                st.error("رمز عبور باید حداقل ۶ کاراکتر باشد")
            else:
                success, error = register(email, password, full_name, org_name)
                if success:
                    st.success("ثبت نام با موفقیت انجام شد")
                    st.rerun()
                else:
                    st.error(f"خطا: {error}")
        
        if back:
            st.session_state.show_register = False
            st.rerun()


# ==================== منوی اصلی ====================

def show_main_menu():
    """نمایش منوی اصلی با توجه به نقش کاربر"""
    
    # دکمه‌های منو در کناری
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📋 منوی اصلی")
    
    # تعریف منوها بر اساس نقش
    menu_items = {
        "operator": ["🏠 صفحه اصلی", "📸 آپلود و پردازش", "📡 پایش زنده", "📊 گزارش‌ها", "⚙️ تنظیمات"],
        "admin": ["🏠 صفحه اصلی", "📸 آپلود و پردازش", "📡 پایش زنده", "📊 گزارش‌ها", "👑 مدیریت", "⚙️ تنظیمات"],
        "viewer": ["🏠 صفحه اصلی", "📡 پایش زنده", "📊 گزارش‌ها"]
    }
    
    user_role = st.session_state.user_role or "operator"
    menus = menu_items.get(user_role, menu_items["operator"])
    
    selected_menu = st.sidebar.radio("", menus, index=0)
    
    # نمایش محتوای صفحه انتخاب شده
    if selected_menu == "🏠 صفحه اصلی":
        show_home_page()
    
    elif selected_menu == "📸 آپلود و پردازش":
        try:
            # ایمپورت پویای صفحات
            from pages import upload
            upload.show_upload_page()
        except ImportError:
            st.info("📤 صفحه آپلود و پردازش در حال آماده‌سازی است")
    
    elif selected_menu == "📡 پایش زنده":
        try:
            from pages import live
            live.show_live_page()
        except ImportError:
            st.info("📡 صفحه پایش زنده در حال آماده‌سازی است")
    
    elif selected_menu == "📊 گزارش‌ها":
        try:
            from pages import reports
            reports.show_reports_page()
        except ImportError:
            st.info("📊 صفحه گزارش‌ها در حال آماده‌سازی است")
    
    elif selected_menu == "👑 مدیریت":
        try:
            from pages import admin
            admin.show_admin_page()
        except ImportError:
            st.info("👑 صفحه مدیریت در حال آماده‌سازی است")
    
    elif selected_menu == "⚙️ تنظیمات":
        try:
            from pages import settings
            settings.show_settings_page()
        except ImportError:
            st.info("⚙️ صفحه تنظیمات در حال آماده‌سازی است")
            # ==================== فوتر ====================

def show_footer():
    """نمایش فوتر صفحه"""
    st.markdown("---")
    st.markdown(f"""
    <div class="main-footer">
        <p>سامانه هوشمند تشخیص پلاک خودروهای ایران | نسخه 3.0</p>
        <p>تمامی حقوق محفوظ است © {datetime.now().year}</p>
        <p style="font-size: 12px;">آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== اجرای اصلی ====================

def main():
    """تابع اصلی برنامه"""
    # مقداردهی اولیه
    init_session_state()
    
    # نمایش هدر
    show_header()
    
    # نمایش اطلاعات کاربر در کناری
    show_user_info()
    
    # بررسی وضعیت احراز هویت
    if st.session_state.token:
        show_main_menu()
    else:
        # نمایش صفحه ثبت نام یا ورود
        if st.session_state.get("show_register", False):
            show_register_page()
        else:
            show_login_page()
    
    # نمایش فوتر
    show_footer()


if __name__ == "__main__":
    main()