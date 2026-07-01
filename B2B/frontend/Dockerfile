# frontend/Dockerfile
# Dockerfile برای سرویس فرانت‌اند استریملیت

FROM python:3.10-slim

# تنظیم متغیرهای محیطی
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ایجاد دایرکتوری کاری
WORKDIR /app

# کپی فایل requirements (در صورت وجود)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir streamlit requests websockets pillow pandas plotly

# کپی کل سورس فرانت‌اند
COPY . .

# ایجاد پوشه‌های لازم (در صورت نیاز)
RUN mkdir -p /app/uploads /app/logs

# expose پورت استریملیت
EXPOSE 8501

# اجرای برنامه
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]