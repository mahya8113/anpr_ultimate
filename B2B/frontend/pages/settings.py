"""
settings.py - صفحه تنظیمات پیشرفته سیستم
قابلیت‌ها: تنظیمات کاربری، تنظیمات تشخیص، تنظیمات اعلان، تنظیمات حریم خصوصی، مدیریت API
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd

# تنظیمات صفحه
st.set_page_config(
    page_title="تنظیمات | سامانه تشخیص پلاک",
    page_icon="⚙️",
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
    .settings-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .settings-card {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .setting-group {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-right: 4px solid #1e3c72;
    }
    .danger-zone {
        border-right: 4px solid #ff4b4b;
        background: #fff5f5;
    }
</style>
""", unsafe_allow_html=True)

# عنوان صفحه
st.markdown("""
<div class="settings-header">
    <h1>⚙️ تنظیمات پیشرفته سیستم</h1>
    <p>پیکربندی تشخیص، اعلان‌ها، حریم خصوصی و مدیریت حساب کاربری</p>
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


def get_user_profile():
    """دریافت اطلاعات کاربر جاری"""
    return api_call("GET", "/auth/me")


def update_user_profile(data):
    """به‌روزرسانی اطلاعات کاربر"""
    return api_call("PUT", "/auth/me", data)


def change_password(old_password, new_password):
    """تغییر رمز عبور"""
    return api_call("POST", "/auth/change-password", {
        "old_password": old_password,
        "new_password": new_password
    })


def get_system_config():
    """دریافت تنظیمات سیستم"""
    return api_call("GET", "/admin/config")


def update_system_config(config):
    """به‌روزرسانی تنظیمات سیستم"""
    return api_call("PUT", "/admin/config", config)


def get_notification_settings():
    """دریافت تنظیمات اعلان کاربر"""
    return api_call("GET", "/notifications/settings")


def update_notification_settings(settings):
    """به‌روزرسانی تنظیمات اعلان"""
    return api_call("PUT", "/notifications/settings", settings)


def get_privacy_settings():
    """دریافت تنظیمات حریم خصوصی سازمان"""
    return api_call("GET", "/admin/privacy-settings")


def update_privacy_settings(settings):
    """به‌روزرسانی تنظیمات حریم خصوصی"""
    return api_call("PUT", "/admin/privacy-settings", settings)


# ==================== منوی کناری ====================

st.sidebar.markdown("## 🔧 دسته‌بندی تنظیمات")

settings_menu = st.sidebar.radio(
    "انتخاب بخش",
    [
        "👤 حساب کاربری",
        "🎯 تنظیمات تشخیص",
        "🔔 اعلان‌ها",
        "🔒 حریم خصوصی",
        "🔑 API Key",
        "⚠️ منطقه خطر"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**کاربر:** {st.session_state.get('user_email', 'نامشخص')}")
st.sidebar.markdown(f"**نقش:** {st.session_state.get('user_role', 'نامشخص')}")


# ==================== 1. حساب کاربری ====================

if settings_menu == "👤 حساب کاربری":
    st.markdown("### 👤 تنظیمات حساب کاربری")
    
    # دریافت اطلاعات کاربر
    user_info, error = get_user_profile()
    
    if error:
        st.error(f"خطا در دریافت اطلاعات: {error}")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form("profile_form"):
                st.markdown("#### 📝 اطلاعات شخصی")
                
                full_name = st.text_input("نام کامل", value=user_info.get("full_name", ""))
                email = st.text_input("ایمیل", value=user_info.get("email", ""))
                phone = st.text_input("شماره تماس", value=user_info.get("phone", ""))
                
                if st.form_submit_button("💾 ذخیره تغییرات"):
                    result, error = update_user_profile({
                        "full_name": full_name,
                        "email": email,
                        "phone": phone
                    })
                    if error:
                        st.error(f"خطا: {error}")
                    else:
                        st.success("اطلاعات با موفقیت به‌روزرسانی شد")
                        st.session_state.user_email = email
        
        with col2:
            with st.form("password_form"):
                st.markdown("#### 🔑 تغییر رمز عبور")
                
                old_password = st.text_input("رمز عبور فعلی", type="password")
                new_password = st.text_input("رمز عبور جدید", type="password")
                confirm_password = st.text_input("تکرار رمز عبور جدید", type="password")
                
                if st.form_submit_button("🔄 تغییر رمز"):
                    if new_password != confirm_password:
                        st.error("رمز عبور جدید و تکرار آن مطابقت ندارند")
                    elif len(new_password) < 6:
                        st.error("رمز عبور جدید باید حداقل ۶ کاراکتر باشد")
                    else:
                        result, error = change_password(old_password, new_password)
                        if error:
                            st.error(f"خطا: {error}")
                        else:
                            st.success("رمز عبور با موفقیت تغییر کرد")
                            st.info("لطفاً دفعه بعد با رمز عبور جدید وارد شوید")


# ==================== 2. تنظیمات تشخیص ====================

elif settings_menu == "🎯 تنظیمات تشخیص":
    st.markdown("### 🎯 تنظیمات پیشرفته تشخیص پلاک")
    
    config, error = get_system_config()
    
    with st.form("detection_settings"):
        st.markdown("#### 🧠 مدل تشخیص")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "آستانه اطمینان تشخیص (%)",
                min_value=0,
                max_value=100,
                value=int((config.get("confidence_threshold", 0.5)) * 100) if config else 50
            )
            
            iou_threshold = st.slider(
                "آستانه IOU (Non-Maximum Suppression)",
                min_value=0,
                max_value=100,
                value=int((config.get("iou_threshold", 0.45)) * 100) if config else 45
            )
        
        with col2:
            max_detections = st.number_input(
                "حداکثر تعداد تشخیص در هر فریم",
                min_value=1,
                max_value=50,
                value=config.get("max_detections", 10) if config else 10
            )
            
            detection_interval = st.number_input(
                "فاصله بین تشخیص‌ها (فریم)",
                min_value=1,
                max_value=60,
                value=config.get("detection_interval", 5) if config else 5
            )
        
        st.markdown("#### 🖼️ پیش‌پردازش تصویر")
        
        col3, col4 = st.columns(2)
        
        with col3:
            enable_clahe = st.checkbox(
                "بهبود کنتراست (CLAHE)",
                value=config.get("enable_clahe", True) if config else True
            )
            
            enable_morphology = st.checkbox(
                "عملیات مورفولوژی (کاهش نویز)",
                value=config.get("enable_morphology", True) if config else True
            )
        
        with col4:
            enable_gamma = st.checkbox(
                "تصحیح گاما",
                value=config.get("enable_gamma", False) if config else False
            )
            
            gamma_value = st.slider(
                "مقدار گاما",
                min_value=0.5,
                max_value=2.5,
                value=config.get("gamma_value", 1.5) if config else 1.5,
                step=0.1,
                disabled=not enable_gamma
            )
        
        st.markdown("#### 🔬 ماژول‌های پیشرفته")
        
        col5, col6 = st.columns(2)
        
        with col5:
            enable_depth = st.checkbox(
                "تخمین عمق (نیاز به GPU)",
                value=config.get("enable_depth", False) if config else False
            )
            
            enable_panoptic = st.checkbox(
                "قطعه‌بندی Panoptic (نیاز به GPU)",
                value=config.get("enable_panoptic", False) if config else False
            )
        
        with col6:
            enable_tracking = st.checkbox(
                "ردیابی چند شیء (DeepSORT)",
                value=config.get("enable_tracking", True) if config else True
            )
            
            enable_anomaly = st.checkbox(
                "تشخیص ناهنجاری",
                value=config.get("enable_anomaly", True) if config else True
            )
        
        if st.form_submit_button("💾 ذخیره تنظیمات تشخیص"):
            new_config = {
                "confidence_threshold": confidence_threshold / 100,
                "iou_threshold": iou_threshold / 100,
                "max_detections": max_detections,
                "detection_interval": detection_interval,
                "enable_clahe": enable_clahe,
                "enable_morphology": enable_morphology,
                "enable_gamma": enable_gamma,
                "gamma_value": gamma_value,
                "enable_depth": enable_depth,
                "enable_panoptic": enable_panoptic,
                "enable_tracking": enable_tracking,
                "enable_anomaly": enable_anomaly
            }
            
            result, error = update_system_config(new_config)
            if error:
                st.error(f"خطا: {error}")
            else:
                st.success("تنظیمات تشخیص با موفقیت ذخیره شد")


# ==================== 3. اعلان‌ها ====================

elif settings_menu == "🔔 اعلان‌ها":
    st.markdown("### 🔔 تنظیمات اعلان‌ها")
    
    notif_settings, error = get_notification_settings()
    
    with st.form("notification_form"):
        st.markdown("#### 📧 ایمیل")
        
        email_enabled = st.checkbox(
            "فعال کردن اعلان ایمیل",
            value=notif_settings.get("email_enabled", True) if notif_settings else True
        )
        
        email_events = st.multiselect(
            "رویدادهای دریافت اعلان ایمیل",
            ["تشخیص پلاک جدید", "ناهنجاری تشخیص داده شد", "ورود خودرو", "خروج خودرو", "خطای سیستمی"],
            default=notif_settings.get("email_events", []) if notif_settings else ["تشخیص پلاک جدید"]
        )
        
        st.markdown("#### 🤖 تلگرام")
        telegram_enabled = st.checkbox(
            "فعال کردن اعلان تلگرام",
            value=notif_settings.get("telegram_enabled", False) if notif_settings else False
        )
        
        telegram_bot_token = st.text_input(
            "توکن ربات تلگرام",
            value=notif_settings.get("telegram_bot_token", "") if notif_settings else "",
            type="password"
        )
        
        telegram_chat_id = st.text_input(
            "Chat ID",
            value=notif_settings.get("telegram_chat_id", "") if notif_settings else ""
        )
        
        st.markdown("#### 🌐 Webhook")
        
        webhook_enabled = st.checkbox(
            "فعال کردن Webhook",
            value=notif_settings.get("webhook_enabled", False) if notif_settings else False
        )
        
        webhook_url = st.text_input(
            "آدرس Webhook",
            value=notif_settings.get("webhook_url", "") if notif_settings else "",
            placeholder="https://your-server.com/webhook"
        )
        
        if st.form_submit_button("💾 ذخیره تنظیمات اعلان"):
            new_settings = {
                "email_enabled": email_enabled,
                "email_events": email_events,
                "telegram_enabled": telegram_enabled,
                "telegram_bot_token": telegram_bot_token,
                "telegram_chat_id": telegram_chat_id,
                "webhook_enabled": webhook_enabled,
                "webhook_url": webhook_url
            }
            
            result, error = update_notification_settings(new_settings)
            if error:
                st.error(f"خطا: {error}")
            else:
                st.success("تنظیمات اعلان با موفقیت ذخیره شد")
                
                if telegram_enabled and telegram_bot_token and telegram_chat_id:
                    st.info("🤖 برای تست ربات، یک پیام آزمایشی ارسال می‌شود")
                    # ارسال پیام تست
                    test_result, _ = api_call("POST", "/notifications/test-telegram")


# ==================== 4. حریم خصوصی ====================

elif settings_menu == "🔒 حریم خصوصی":
    st.markdown("### 🔒 تنظیمات حریم خصوصی و نگهداری داده")
    
    privacy, error = get_privacy_settings()
    
    with st.form("privacy_form"):
        st.markdown("#### 📅 نگهداری داده‌ها")
        
        retention_days_raw = st.number_input(
            "مدت نگهداری تصاویر خام (روز)",
            min_value=0,
            max_value=365,
            value=privacy.get("raw_image_retention_days", 30) if privacy else 30,
            help="۰ به معنی حذف فوری تصاویر است"
        )
        
        retention_days_plate = st.number_input(
            "مدت نگهداری متن پلاک (روز)",
            min_value=0,
            max_value=730,
            value=privacy.get("plate_text_retention_days", 365) if privacy else 365,
            help="پس از این مدت، متن پلاک با هش جایگزین می‌شود"
        )
        
        st.markdown("#### 🎭 ناشناس‌سازی")
        
        anonymize_faces = st.checkbox(
            "محو کردن خودکار چهره رانندگان",
            value=privacy.get("anonymize_faces", True) if privacy else True
        )
        
        anonymize_plates = st.checkbox(
            "محو کردن پلاک‌های سایر خودروها (غیر از خودرو هدف)",
            value=privacy.get("anonymize_other_plates", True) if privacy else True
        )
        
        st.markdown("#### 📊 اشتراک‌گذاری داده")
        
        share_analytics = st.checkbox(
            "ارسال آمار ناشناس برای بهبود سیستم",
            value=privacy.get("share_anonymous_analytics", False) if privacy else False,
            help="این داده‌ها شامل اطلاعات شخصی نیست و فقط برای بهبود دقت مدل استفاده می‌شود"
        )
        
        if st.form_submit_button("💾 ذخیره تنظیمات حریم خصوصی"):
            new_settings = telegram_enabled = st.checkbox(
            "فعال کردن اعلان تلگرام",
            value=notif_settings.get("telegram_enabled", False) if notif_settings else False
        )
        
        telegram_bot_token = st.text_input(
            "توکن ربات تلگرام",
            value=notif_settings.get("telegram_bot_token", "") if notif_settings else "",
            type="password"
        )
        
        telegram_chat_id = st.text_input(
            "Chat ID",
            value=notif_settings.get("telegram_chat_id", "") if notif_settings else ""
        )
        
        st.markdown("#### 🌐 Webhook")
        
        webhook_enabled = st.checkbox(
            "فعال کردن Webhook",
            value=notif_settings.get("webhook_enabled", False) if notif_settings else False
        )
        
        webhook_url = st.text_input(
            "آدرس Webhook",
            value=notif_settings.get("webhook_url", "") if notif_settings else "",
            placeholder="https://your-server.com/webhook"
        )
        
        if st.form_submit_button("💾 ذخیره تنظیمات اعلان"):
            new_settings = {
                "email_enabled": email_enabled,
                "email_events": email_events,
                "telegram_enabled": telegram_enabled,
                "telegram_bot_token": telegram_bot_token,
                "telegram_chat_id": telegram_chat_id,
                "webhook_enabled": webhook_enabled,
                "webhook_url": webhook_url
            }
            
            result, error = update_notification_settings(new_settings)
            if error:
                st.error(f"خطا: {error}")
            else:
                st.success("تنظیمات اعلان با موفقیت ذخیره شد")
                
                if telegram_enabled and telegram_bot_token and telegram_chat_id:
                    st.info("🤖 برای تست ربات، یک پیام آزمایشی ارسال می‌شود")
                    # ارسال پیام تست
                    test_result, _ = api_call("POST", "/notifications/test-telegram")


# ==================== 4. حریم خصوصی ====================

elif settings_menu == "🔒 حریم خصوصی":
    st.markdown("### 🔒 تنظیمات حریم خصوصی و نگهداری داده")
    
    privacy, error = get_privacy_settings()
    
    with st.form("privacy_form"):
        st.markdown("#### 📅 نگهداری داده‌ها")
        
        retention_days_raw = st.number_input(
            "مدت نگهداری تصاویر خام (روز)",
            min_value=0,
            max_value=365,
            value=privacy.get("raw_image_retention_days", 30) if privacy else 30,
            help="۰ به معنی حذف فوری تصاویر است"
        )
        
        retention_days_plate = st.number_input(
            "مدت نگهداری متن پلاک (روز)",
            min_value=0,
            max_value=730,
            value=privacy.get("plate_text_retention_days", 365) if privacy else 365,
            help="پس از این مدت، متن پلاک با هش جایگزین می‌شود"
        )
        
        st.markdown("#### 🎭 ناشناس‌سازی")
        
        anonymize_faces = st.checkbox(
            "محو کردن خودکار چهره رانندگان",
            value=privacy.get("anonymize_faces", True) if privacy else True
        )
        
        anonymize_plates = st.checkbox(
            "محو کردن پلاک‌های سایر خودروها (غیر از خودرو هدف)",
            value=privacy.get("anonymize_other_plates", True) if privacy else True
        )
        
        st.markdown("#### 📊 اشتراک‌گذاری داده")
        
        share_analytics = st.checkbox(
            "ارسال آمار ناشناس برای بهبود سیستم",
            value=privacy.get("share_anonymous_analytics", False) if privacy else False,
            help="این داده‌ها شامل اطلاعات شخصی نیست و فقط برای بهبود دقت مدل استفاده می‌شود"
        )
        
        if st.form_submit_button("💾 ذخیره تنظیمات حریم خصوصی"):
            new_settings ={
                "raw_image_retention_days": retention_days_raw,
                "plate_text_retention_days": retention_days_plate,
                "anonymize_faces": anonymize_faces,
                "anonymize_other_plates": anonymize_plates,
                "share_anonymous_analytics": share_analytics
            }
            
            result, error = update_privacy_settings(new_settings)
            if error:
                st.error(f"خطا: {error}")
            else:
                st.success("تنظیمات حریم خصوصی با موفقیت ذخیره شد")
                
                if retention_days_raw == 0:
                    st.warning("⚠️ توجه: تصاویر خام بلافاصله پس از پردازش حذف خواهند شد")


# ==================== 5. API Key ====================

elif settings_menu == "🔑 API Key":
    st.markdown("### 🔑 مدیریت API Key")
    
    tab1, tab2 = st.tabs(["📋 کلیدهای فعال", "➕ ایجاد کلید جدید"])
    
    with tab1:
        st.markdown("#### کلیدهای API فعال")
        
        api_keys, error = api_call("GET", "/api-keys")
        
        if error:
            st.error(f"خطا: {error}")
        elif api_keys:
            df = pd.DataFrame(api_keys)
            st.dataframe(df, use_container_width=True)
            
            for key in api_keys:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.code(key.get("key_preview", "••••"), language="text")
                with col2:
                    st.write(f" expires: {key.get('expires_at', 'نامحدود')}")
                with col3:
                    if st.button(f"🗑️ حذف", key=f"del_{key['id']}"):
                        result, error = api_call("DELETE", f"/api-keys/{key['id']}")
                        if error:
                            st.error(f"خطا: {error}")
                        else:
                            st.success("کلید حذف شد")
                            st.rerun()
        else:
            st.info("هیچ کلید API فعالی یافت نشد")
    
    with tab2:
        with st.form("create_api_key"):
            st.markdown("#### ایجاد کلید API جدید")
            
            key_name = st.text_input("نام کلید", placeholder="مثال: API_KEY_پارکینگ_مرکزی")
            key_expires_days = st.number_input("مدت اعتبار (روز)", min_value=1, max_value=365, value=30)
            key_permissions = st.multiselect(
                "دسترسی‌ها",
                ["detect", "track", "reports", "admin"],
                default=["detect"]
            )
            
            if st.form_submit_button("🔑 ایجاد کلید جدید"):
                data = {
                    "name": key_name,
                    "expires_days": key_expires_days,
                    "permissions": key_permissions
                }
                result, error = api_call("POST", "/api-keys", data)
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.success("کلید API با موفقیت ایجاد شد")
                    st.markdown("**کلید جدید:**")
                    st.code(result.get("api_key", ""), language="text")
                    st.warning("⚠️ لطفاً کلید را در جای امن ذخیره کنید. پس از خروج از این صفحه، دیگر نمایش داده نخواهد شد")


# ==================== 6. منطقه خطر ====================

elif settings_menu == "⚠️ منطقه خطر":
    st.markdown("### ⚠️ منطقه خطر")
    
    st.warning("""
    ⚠️ هشدار: تغییرات در این بخش غیرقابل بازگشت است. لطفاً با دقت عمل کنید.
    """)
    
    with st.expander("🗑️ حذف تمام داده‌های تشخیص", expanded=False):
        st.markdown("""
        این عملیات تمام داده‌های تشخیص پلاک را حذف می‌کند.
        این عملیات غیرقابل بازگشت است.
        """)
        
        confirm_text = st.text_input("برای تأیید، عبارت 'حذف داده‌ها' را وارد کنید")
        
        if st.button("🗑️ حذف همه داده‌ها", use_container_width=True):
            if confirm_text == "حذف داده‌ها":
                result, error = api_call("DELETE", "/admin/delete-all-detections")
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.success("تمام داده‌های تشخیص حذف شد")
            else:
                st.error("عبارت تأیید نادرست است")
    
    with st.expander("🔄 بازنشانی تنظیمات به حالت پیش‌فرض", expanded=False):
        st.markdown("تمامی تنظیمات سیستم به حالت اولیه بازمی‌گردد.")
        
        if st.button("🔄 بازنشانی تنظیمات", use_container_width=True):
            result, error = api_call("POST", "/admin/reset-settings")
            if error:
                st.error(f"خطا: {error}")
            else:
                st.success("تنظیمات به حالت پیش‌فرض بازگردانده شد")
    
    with st.expander("🚫 غیرفعال کردن حساب کاربری", expanded=False):
        st.markdown("حساب کاربری شما غیرفعال می‌شود و دیگر نمی‌توانید وارد شوید.")
        
        password_confirm = st.text_input("رمز عبور خود را وارد کنید", type="password")
        
        if st.button("🚫 غیرفعال کردن حساب", use_container_width=True):
            if password_confirm:
                result, error = api_call("POST", "/auth/deactivate", {"password": password_confirm})
                if error:
                    st.error(f"خطا: {error}")
                else:
                    st.error("حساب کاربری شما غیرفعال شد. در حال خروج...")
                    st.session_state.clear()
                    st.rerun()
            else:
                st.error("لطفاً رمز عبور را وارد کنید")


# ==================== فوتر ====================

st.markdown("---")
st.caption(f"آخرین تغییرات: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            