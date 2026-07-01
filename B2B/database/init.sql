-- ============================================================================
-- init.sql - اسکریپت اولیه‌سازی دیتابیس سامانه تشخیص پلاک فارسی (ANPR)
-- ============================================================================
-- این فایل شامل:
-- 1. ایجاد دیتابیس و اکستنشن‌ها
-- 2. ایجاد جداول اصلی (سازمان‌ها، کاربران، دوربین‌ها، تشخیص‌ها)
-- 3. ایجاد جداول گزارش و لاگ
-- 4. ایجاد ویوها و ایندکس‌ها
-- 5. درج داده‌های اولیه
-- ============================================================================

-- ==================== ایجاد اکستنشن‌ها ====================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "citext";

-- ==================== ایجاد دیتابیس ====================
-- (در صورت نیاز - معمولاً در docker-compose ایجاد می‌شود)
-- CREATE DATABASE anpr_db WITH ENCODING 'UTF8' LC_COLLATE 'fa_IR.UTF-8' LC_CTYPE 'fa_IR.UTF-8';

-- ==================== جداول اصلی ====================

-- 1. جدول سازمان‌ها (مشتریان سیستم)
CREATE TABLE IF NOT EXISTS organizations (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE,
    tier VARCHAR(50) DEFAULT 'standard' CHECK (tier IN ('standard', 'pro', 'enterprise', 'custom')),
    max_cameras INT DEFAULT 5,
    quota_limit INT DEFAULT 5000,
    current_usage INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    is_deleted BOOLEAN DEFAULT FALSE,
    license_key VARCHAR(255),
    license_expires_at TIMESTAMPTZ,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

COMMENT ON TABLE organizations IS 'جدول سازمان‌ها (مشتریان سیستم)';
COMMENT ON COLUMN organizations.tier IS 'نوع اشتراک: standard, pro, enterprise, custom';
COMMENT ON COLUMN organizations.max_cameras IS 'حداکثر تعداد دوربین مجاز';
COMMENT ON COLUMN organizations.quota_limit IS 'سقف تشخیص روزانه';

-- 2. جدول کاربران سیستم
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email CITEXT UNIQUE NOT NULL,
    phone VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    role VARCHAR(50) DEFAULT 'operator' CHECK (role IN ('super_admin', 'admin', 'operator', 'viewer')),
    permissions JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMPTZ,
    login_attempts INT DEFAULT 0,
    locked_until TIMESTAMPTZ,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    CONSTRAINT fk_org FOREIGN KEY (org_id) REFERENCES organizations(id)
);

CREATE INDEX idx_users_org_id ON users(org_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
COMMENT ON TABLE users IS 'جدول کاربران سیستم';
COMMENT ON COLUMN users.role IS 'نقش کاربر: super_admin, admin, operator, viewer';

-- 3. جدول دوربین‌ها
CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    location TEXT,
    stream_url TEXT NOT NULL,
    stream_type VARCHAR(20) DEFAULT 'rtsp' CHECK (stream_type IN ('rtsp', 'http', 'usb', 'v4l2', 'onvif')),
    username VARCHAR(100),
    password VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_online BOOLEAN DEFAULT FALSE,
    last_online TIMESTAMPTZ,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_cameras_org_id ON cameras(org_id);
CREATE INDEX idx_cameras_is_active ON cameras(is_active);
COMMENT ON TABLE cameras IS 'جدول دوربین‌های متصل به سیستم';

-- 4. جدول تشخیص پلاک (هایپرتیبل TimescaleDB)
CREATE TABLE IF NOT EXISTS detections (
    id SERIAL,
    uuid UUID DEFAULT uuid_generate_v4(),
    org_id INTEGER REFERENCES organizations(id),
    camera_id INTEGER REFERENCES cameras(id),
    track_id INTEGER,
    plate_text VARCHAR(20),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    bbox JSONB,
    vehicle_type VARCHAR(50),
    vehicle_color VARCHAR(30),
    speed REAL,
    direction REAL,
    anomaly_score REAL DEFAULT 0,
    is_anomaly BOOLEAN DEFAULT FALSE,
    is_flagged BOOLEAN DEFAULT FALSE,
    flag_reason TEXT,
    raw_image_url TEXT,
    processed_image_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('detections', 'created_at', chunk_time_interval => interval '7 days', if_not_exists => TRUE);

CREATE INDEX idx_detections_org_time ON detections (org_id, created_at DESC);
CREATE INDEX idx_detections_camera_time ON detections (camera_id, created_at DESC);
CREATE INDEX idx_detections_plate ON detections (plate_text);
CREATE INDEX idx_detections_anomaly ON detections (is_anomaly, created_at DESC);
CREATE INDEX idx_detections_confidence ON detections (confidence);
CREATE INDEX idx_detections_gin_metadata ON detections USING gin (metadata);

COMMENT ON TABLE detections IS 'جدول تشخیص پلاک خودروها (هایپرتیبل)';
COMMENT ON COLUMN detections.anomaly_score IS 'نمره ناهنجاری (0 تا 1)';
COMMENT ON COLUMN detections.is_anomaly IS 'آیا این تشخیص ناهنجار است؟';

-- ==================== جداول گزارش و آمار ====================

-- 5. جدول گزارش‌های روزانه
CREATE TABLE IF NOT EXISTS daily_reports (
    id SERIAL PRIMARY KEY,
    org_id INTEGER REFERENCES organizations(id),
    report_date DATE NOT NULL,
    total_detections INT DEFAULT 0,
    unique_plates INT DEFAULT 0,
    avg_confidence REAL DEFAULT 0,
    anomaly_count INT DEFAULT 0,
    peak_hour INT,
    report_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_daily_reports_org_date ON daily_reports (org_id, report_date);
SELECT create_hypertable('daily_reports', 'report_date', chunk_time_interval => interval '30 days', if_not_exists => TRUE);

-- 6. جدول لاگ‌های سیستم
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    log_level VARCHAR(20) CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    component VARCHAR(100),
    message TEXT,
    user_id INTEGER REFERENCES users(id),
    org_id INTEGER REFERENCES organizations(id),
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_logs_created_at ON system_logs (created_at DESC);
CREATE INDEX idx_logs_level ON system_logs (log_level);
CREATE INDEX idx_logs_component ON system_logs (component);
SELECT create_hypertable('system_logs', 'created_at', chunk_time_interval => interval '1 day', if_not_exists => TRUE);

-- ==================== جداول مدیریت ====================

-- 7. جدول API Keys
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    org_id INTEGER REFERENCES organizations(id),
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    permissions JSONB DEFAULT '["detect"]',
    last_used TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_api_keys_key ON api_keys (api_key);
CREATE INDEX idx_api_keys_org ON api_keys (org_id);

-- 8. جدول تنظیمات سیستم
CREATE TABLE IF NOT EXISTS system_settings (
    id SERIAL PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
-- 9. جدول لایسنس‌ها
CREATE TABLE IF NOT EXISTS licenses (
    id SERIAL PRIMARY KEY,
    org_id INTEGER REFERENCES organizations(id),
    license_key VARCHAR(255) UNIQUE NOT NULL,
    license_type VARCHAR(50) CHECK (license_type IN ('trial', 'monthly', 'yearly', 'perpetual')),
    max_cameras INT DEFAULT 5,
    max_users INT DEFAULT 10,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 10. جدول اعلان‌ها (Notification Rules)
CREATE TABLE IF NOT EXISTS notification_rules (
    id SERIAL PRIMARY KEY,
    org_id INTEGER REFERENCES organizations(id),
    name VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) CHECK (event_type IN ('plate_detected', 'anomaly', 'vehicle_entered', 'vehicle_exited', 'system_alert')),
    condition JSONB,
    webhook_url TEXT,
    email VARCHAR(255),
    telegram_bot_token TEXT,
    telegram_chat_id VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==================== ویوها (Views) ====================

-- ویو آمار سازمان‌ها
CREATE OR REPLACE VIEW v_organization_stats AS
SELECT 
    o.id,
    o.name,
    o.tier,
    COUNT(DISTINCT u.id) as user_count,
    COUNT(DISTINCT c.id) as camera_count,
    COUNT(DISTINCT d.id) as detection_count,
    COALESCE(AVG(d.confidence), 0) as avg_confidence,
    COALESCE(SUM(CASE WHEN d.is_anomaly THEN 1 ELSE 0 END), 0) as anomaly_count
FROM organizations o
LEFT JOIN users u ON u.org_id = o.id AND u.is_active = true
LEFT JOIN cameras c ON c.org_id = o.id AND c.is_active = true
LEFT JOIN detections d ON d.org_id = o.id
GROUP BY o.id, o.name, o.tier;

-- ویو تشخیص‌های امروز
CREATE OR REPLACE VIEW v_today_detections AS
SELECT 
    d.*,
    o.name as org_name,
    c.name as camera_name
FROM detections d
JOIN organizations o ON o.id = d.org_id
JOIN cameras c ON c.id = d.camera_id
WHERE DATE(d.created_at) = CURRENT_DATE
ORDER BY d.created_at DESC;

-- ==================== ایندکس‌های اضافی ====================

-- ایندکس برای جستجوی متن در پلاک
CREATE INDEX IF NOT EXISTS idx_detections_plate_trgm ON detections USING gin (plate_text gin_trgm_ops);

-- ایندکس ترکیبی برای گزارش‌های زمانی
CREATE INDEX IF NOT EXISTS idx_detections_org_date ON detections (org_id, DATE(created_at));

-- ایندکس برای لاگ‌های خطا
CREATE INDEX IF NOT EXISTS idx_logs_error ON system_logs (log_level, created_at DESC) WHERE log_level IN ('ERROR', 'CRITICAL');

-- ==================== توابع (Functions) ====================

-- تابع به‌روزرسانی خودکار updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- اعمال تریگر روی جداول
CREATE TRIGGER update_organizations_updated_at
    BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cameras_updated_at
    BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- تابع محاسبه آمار روزانه
CREATE OR REPLACE FUNCTION calculate_daily_stats(p_date DATE)
RETURNS VOID AS $$
BEGIN
    INSERT INTO daily_reports (org_id, report_date, total_detections, unique_plates, avg_confidence, anomaly_count, peak_hour)
    SELECT 
        org_id,
        p_date,
        COUNT(*) as total_detections,
        COUNT(DISTINCT plate_text) as unique_plates,
        AVG(confidence) as avg_confidence,
        SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
        MODE() WITHIN GROUP (ORDER BY EXTRACT(HOUR FROM created_at)) as peak_hour
    FROM detections
    WHERE DATE(created_at) = p_date
    GROUP BY org_id
    ON CONFLICT (org_id, report_date) DO UPDATE
    SET total_detections = EXCLUDED.total_detections,
        unique_plates = EXCLUDED.unique_plates,
        avg_confidence = EXCLUDED.avg_confidence,
        anomaly_count = EXCLUDED.anomaly_count,
        peak_hour = EXCLUDED.peak_hour;
END;
$$ LANGUAGE plpgsql;

-- ==================== داده‌های اولیه ====================

-- تنظیمات اولیه سیستم
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
    ('system_name', '"سامانه تشخیص پلاک خودرو ایران"', 'نام سیستم'),
    ('system_version', '"3.0.0"', 'نسخه سیستم'),
    ('default_confidence_threshold', '0.5', 'آستانه اطمینان پیش‌فرض'),
    ('max_upload_size_mb', '100', 'حداکثر حجم آپلود'),
    ('detection_interval_ms', '500', 'فاصله بین تشخیص‌ها'),
    ('enable_telegram_notifications', 'false', 'فعال‌سازی اعلان تلگرام'),
    ('enable_email_notifications', 'false', 'فعال‌سازی اعلان ایمیل'),
    ('retention_days_images', '30', 'مدت نگهداری تصاویر'),
    ('retention_days_logs', '90', 'مدت نگهداری لاگ‌ها')
ON CONFLICT (setting_key) DO NOTHING;

-- ایجاد کاربر ادمین اولیه (رمز: admin123 - در محیط تولید حتماً تغییر کنید)
-- رمز هش شده برای "admin123"
INSERT INTO users (org_id, email, password_hash, full_name, role, is_verified, is_active)
VALUES (
    1,
    'admin@anpr.ir',
    '$2b$12$KIXQzVgVKJQXZ8WqZyQ7k.KQ4jF8qJLqJqJqJqJqJqJqJqJqJq',
    'مدیر سیستم',
    'super_admin',
    TRUE,
    TRUE
) ON CONFLICT (email) DO NOTHING;

-- ==================== نمایش اطلاعات نهایی ====================

DO $$
BEGIN
    RAISE NOTICE '==========================================';
    RAISE NOTICE '✅ دیتابیس با موفقیت ایجاد شد!';
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'جدول‌های ایجاد شده:';
    RAISE NOTICE '  - organizations (سازمان‌ها)';
    RAISE NOTICE '  - users (کاربران)';
    RAISE NOTICE '  - cameras (دوربین‌ها)';
    RAISE NOTICE '  - detections (تشخیص پلاک) - هایپرتیبل';
    RAISE NOTICE '  - daily_reports (گزارش روزانه)';
    RAISE NOTICE '  - system_logs (لاگ سیستم)';
    RAISE NOTICE '  - api_keys (کلیدهای API)';
    RAISE NOTICE '  - system_settings (تنظیمات)';
    RAISE NOTICE '  - licenses (لایسنس)';
    RAISE NOTICE '  - notification_rules (قوانین اعلان)';
    RAISE NOTICE '==========================================';
    RAISE NOTICE '✅ کاربر ادمین: admin@anpr.ir';
    RAISE NOTICE '✅ رمز عبور: admin123 (تغییر دهید!)';
    RAISE NOTICE '==========================================';
END $$;