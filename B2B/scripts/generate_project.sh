#!/bin/bash
# generate_project.sh
# اسکریپت خودکار تولید کامل پروژه ANPR از صفر تا صد
# اجرا: chmod +x generate_project.sh && ./generate_project.sh

set -e  # توقف در صورت خطا

# ==================== رنگ‌ها ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==================== توابع ====================
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# ==================== شروع پروژه ====================
print_header "🚗 تولید پروژه کامل ANPR (تشخیص پلاک فارسی)"

# دریافت نام پروژه
PROJECT_NAME=${1:-"anpr-ultimate"}
print_info "نام پروژه: $PROJECT_NAME"

# ایجاد پوشه اصلی
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"
print_status "پوشه اصلی $PROJECT_NAME ایجاد شد"

# ==================== ایجاد ساختار پوشه‌ها ====================
print_header "📁 ایجاد ساختار پوشه‌ها"

mkdir -p backend/{core,modules,api/{routes,websocket},services,utils}
mkdir -p frontend/{pages,static}
mkdir -p tests
mkdir -p scripts
mkdir -p models
mkdir -p database
mkdir -p monitoring
mkdir -p mlflow
mkdir -p kubernetes
mkdir -p secrets
mkdir -p traefik
mkdir -p docs

print_status "ساختار پوشه‌ها ایجاد شد"

# ==================== ایجاد فایل‌های اصلی ====================
print_header "📝 ایجاد فایل‌های اصلی"

# .env
cat > .env << 'EOF'
# ANPR Environment Variables
POSTGRES_USER=anpr
POSTGRES_PASSWORD=anpr_strong_pwd
POSTGRES_DB=anpr_db
POSTGRES_HOST=postgres

REDIS_PASSWORD=redis_strong_pwd
REDIS_URL=redis://default:redis_strong_pwd@redis:6379/0

RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_ENDPOINT=minio:9000

JWT_SECRET=super-secret-production-key-change-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

LICENSE_SECRET=your_licence_secret_change_me

MODEL_PATH=/app/models/yolov8n.pt
CRNN_MODEL_PATH=/app/models/crnn_persian_v1.pth
CONF_THRESH=0.5
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

ZARINPAL_MERCHANT=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
ZARINPAL_CALLBACK_URL=https://anpr.ir/payment/callback

SENTRY_DSN=
EOF
print_status ".env ایجاد شد"

# .gitignore
cat > .gitignore << 'EOF'
# Python
pycache/
*.py[cod]
*.so
.Python
env/
venv/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env
secrets/
models/*.pt
models/*.pth
models/*.onnx
models/*.engine
mlflow/
logs/
*.log
.DS_Store
*.db
*.sqlite3

# Docker
*.pid

# Testing
.pytest_cache/
.coverage
htmlcov/

# Uploads
uploads/
temp/
EOF
print_status ".gitignore ایجاد شد"

# docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

networks:
  anpr-net:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
  minio_data:
  model_data:

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: anpr
      POSTGRES_PASSWORD: anpr_strong_pwd
      POSTGRES_DB: anpr_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - anpr-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U anpr"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass redis_strong_pwd
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - anpr-net
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
      rabbitmq:
    image: rabbitmq:3.12-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    networks:
      - anpr-net

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    networks:
      - anpr-net

  backend:
    build: ./backend
    environment:
      POSTGRES_HOST: postgres
      REDIS_URL: ${REDIS_URL}
      RABBITMQ_URL: ${RABBITMQ_URL}
      JWT_SECRET: ${JWT_SECRET}
      LICENSE_SECRET: ${LICENSE_SECRET}
      MODEL_PATH: /app/models/yolov8n.pt
    volumes:
      - ./models:/app/models
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - rabbitmq
      - minio
    networks:
      - anpr-net
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
    networks:
      - anpr-net
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - anpr-net
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    ports:
      - "3000:3000"
    networks:
      - anpr-net
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
  minio_data:
  model_data:
EOF
print_status "docker-compose.yml ایجاد شد"

# Makefile
cat > Makefile << 'EOF'
.PHONY: help up down build logs clean test train

help:
	@echo "Available commands:"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make build       - Build Docker images"
	@echo "  make logs        - Show logs"
	@echo "  make clean       - Remove all containers and volumes"
	@echo "  make test        - Run tests"
	@echo "  make train       - Train YOLO model"
	@echo "  make download    - Download datasets"

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build --no-cache

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf postgres_data redis_data rabbitmq_data minio_data model_data

test:
	pytest tests/ -v

train:
	python scripts/train_yolo.py --data dataset/data.yaml --epochs 50

download:
	python scripts/download_datasets.py --all

dev:
	docker-compose up
EOF
print_status "Makefile ایجاد شد"

# README.md
cat > README.md << 'EOF'
# 🚗 ANPR Ultimate - سامانه تشخیص پلاک خودروهای ایران

## قابلیت‌ها
- تشخیص پلاک خودرو با دقت بالا
- پشتیبانی از تصاویر و ویدئو
- پایش زنده دوربین‌ها
- گزارش‌گیری پیشرفته (PDF/Excel)
- پنل مدیریت سازمانی
- اعلان‌های خودکار (ایمیل/تلگرام/Webhook)
- مانیتورینگ کامل با Prometheus + Grafana

## راه‌اندازی سریع

### پیش‌نیازها
- Docker و Docker Compose
- حداقل 4GB RAM

### نصب و اجرا
`bash
# 1. کلون پروژه
git clone <your-repo>
cd anpr-ultimate

# 2. اجرا با docker-compose
docker-compose up -d

# 3. دسترسی به سامانه
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090