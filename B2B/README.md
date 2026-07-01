# 🚗 ANPR Ultimate - سامانه هوشمند تشخیص پلاک خودروهای ایران

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-24.0+-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/license-Apache%202.0-red.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

<p align="center">
  <img src="docs/images/anpr-logo.png" alt="ANPR Logo" width="200">
</p>

<p align="center">
  <strong>سیستم پیشرفته تشخیص و خواندن پلاک خودروهای ایرانی با قابلیت‌های هوش مصنوعی</strong>
</p>

<p align="center">
  <a href="#ویژگی‌ها">ویژگی‌ها</a> •
  <a href="#تکنولوژی‌ها">تکنولوژی‌ها</a> •
  <a href="#شروع-سریع">شروع سریع</a> •
  <a href="#ساختار-پروژه">ساختار</a> •
  <a href="#api-مستندات">API</a> •
  <a href="#فروش-و-تجاری-سازی">فروش</a> •
  <a href="#پشتیبانی">پشتیبانی</a>
</p>

---

## ✨ ویژگی‌ها

### 🎯 قابلیت‌های اصلی
- تشخیص خودکار پلاک خودرو با دقت بالای 95% در شرایط نوری مختلف
- پشتیبانی از تصاویر و ویدئو با فرمت‌های JPEG, PNG, MP4, AVI, MOV
- پایش زنده دوربین‌ها با قابلیت تشخیص لحظه‌ای
- OCR پیشرفته فارسی برای خواندن حروف و اعداد پلاک ایران
- گزارش‌گیری پیشرفته با خروجی PDF، Excel و CSV
- پنل مدیریت سازمانی کامل برای ادمین‌ها
- اعلان‌های خودکار از طریق ایمیل، تلگرام و Webhook

### 🧠 قابلیت‌های هوش مصنوعی
- تشخیص ناهنجاری (حرکت خلاف جهت، سرعت غیرمجاز، توقف طولانی)
- ردیابی چند شیء (Multi-Object Tracking) با DeepSORT
- تخمین عمق (Depth Estimation) برای صحنه‌های سه بعدی
- قطعه‌بندی Panoptic برای درک کامل صحنه
- بهبود تصاویر در نور کم با Zero-DCE
- تطابق دامنه و تولید داده مصنوعی با GAN
- هوش مصنوعی قابل تفسیر (XAI) با GradCAM و SHAP

### 🔒 امنیت و مدیریت
- احراز هویت JWT با قابلیت Refresh Token
- سیستم لایسنس برای مدیریت سازمان‌ها
- نرخ محدودیت (Rate Limiting) برای جلوگیری از حملات
- حریم خصوصی با قابلیت محو خودکار چهره
- مانیتورینگ کامل با Prometheus و Grafana

### ☁️ استقرار و مقیاس‌پذیری
- استقرار روی Kubernetes با Helm Charts
- مقیاس‌پذیری خودکار با HPA
- پشتیبانی از Docker Compose برای توسعه
- SSL خودکار با Let's Encrypt و Traefik
- قابلیت اجرا روی Edge devices با ONNX و TensorRT

---

## 🛠 تکنولوژی‌ها

| دسته | تکنولوژی |
|------|-----------|
| Backend | FastAPI, Python 3.10 |
| Frontend | Streamlit, React (Admin Panel) |
| Database | PostgreSQL, TimescaleDB |
| Cache | Redis |
| Message Queue | RabbitMQ |
| Storage | MinIO (S3-compatible) |
| ML/DL | PyTorch, YOLOv8, CRNN, LSTM, GAN |
| OCR | EasyOCR, CRNN (فارسی) |
| Monitoring | Prometheus, Grafana |
| Container | Docker, Docker Compose |
| Orchestration | Kubernetes, Helm |
| Proxy | Traefik, Nginx |
| CI/CD | GitHub Actions |

---

## 🚀 شروع سریع

### پیش‌نیازها

- Docker (24.0+) و Docker Compose (2.20+)
- Python 3.10+ (برای توسعه)
- 8GB RAM (حداقل)
- GPU NVIDIA (اختیاری - برای تشخیص سریع‌تر)

### نصب و اجرا

#### روش 1: استفاده از Docker Compose (سریع)

`bash
# 1. کلون پروژه
git clone https://github.com/your-repo/anpr-ultimate.git
cd anpr-ultimate

# 2. کپی فایل محیطی
cp .env.example .env
# فایل .env را با مقادیر واقعی ویرایش کنید

# 3. اجرای سرویس‌ها
docker-compose up -d

# 4. مشاهده وضعیت
docker-compose ps

# 5. دسترسی به سامانه
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/admin)
