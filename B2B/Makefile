# ============================================================================
# Makefile - ابزار خودکارسازی پروژه ANPR (سامانه تشخیص پلاک فارسی)
# ============================================================================
# دستورات موجود:
#   make help      - نمایش راهنما
#   make up        - اجرای سرویس‌ها با docker-compose
#   make down      - توقف سرویس‌ها
#   make build     - ساخت مجدد تصاویر داکر
#   make logs      - نمایش لاگ‌های سرویس‌ها
#   make clean     - پاکسازی کامل (حذف کانتینرها و حجم‌ها)
#   make test      - اجرای تست‌ها
#   make test-cov  - اجرای تست‌ها با گزارش کاور
#   make lint      - بررسی کیفیت کد
#   make format    - فرمت کردن کد
#   make install   - نصب وابستگی‌ها
#   make train     - آموزش مدل YOLO
#   make train-crnn- آموزش مدل CRNN
#   make download  - دانلود دیتاست‌ها
#   make export    - تبدیل مدل به ONNX
#   make migrate   - اجرای مهاجرت دیتابیس
#   make backup    - بک‌آپ گرفتن از دیتابیس
#   make restore   - بازیابی دیتابیس
#   make shell     - ورود به شل بک‌اند
#   make ps        - مشاهده وضعیت سرویس‌ها
#   make restart   - ریستارت سرویس‌ها
#   make logs-api  - نمایش لاگ بک‌اند
#   make logs-front- نمایش لاگ فرانت‌اند
#   make status    - وضعیت پروژه
#   make deploy    - استقرار روی سرور
# ============================================================================

# ==================== متغیرها ====================
PROJECT_NAME := anpr
DOCKER_COMPOSE := docker-compose
PYTHON := python3
PIP := pip3
POETRY := poetry

# رنگ‌ها برای خروجی
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ==================== اهداف اصلی ====================
.PHONY: help
help: ## نمایش راهنمای دستورات
	@echo "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
	@echo "${BLUE}║     Makefile - سامانه تشخیص پلاک فارسی (ANPR)                ║${NC}"
	@echo "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
	@echo ""
	@echo "${GREEN}دستورات موجود:${NC}"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${YELLOW}%-20s${NC} %s\n", $$1, $$2}'
	@echo ""

# ==================== Docker Commands ====================
.PHONY: up
up: ## اجرای سرویس‌ها با docker-compose
	@echo "${GREEN}▶ راه‌اندازی سرویس‌ها...${NC}"
	$(DOCKER_COMPOSE) up -d
	@echo "${GREEN}✅ سرویس‌ها با موفقیت راه‌اندازی شدند${NC}"
	@$(DOCKER_COMPOSE) ps

.PHONY: down
down: ## توقف سرویس‌ها
	@echo "${YELLOW}■ توقف سرویس‌ها...${NC}"
	$(DOCKER_COMPOSE) down
	@echo "${GREEN}✅ سرویس‌ها متوقف شدند${NC}"

.PHONY: build
build: ## ساخت مجدد تصاویر داکر
	@echo "${BLUE}🔨 ساخت تصاویر داکر...${NC}"
	$(DOCKER_COMPOSE) build --no-cache
	@echo "${GREEN}✅ تصاویر با موفقیت ساخته شدند${NC}"

.PHONY: logs
logs: ## نمایش لاگ‌های همه سرویس‌ها
	$(DOCKER_COMPOSE) logs -f

.PHONY: logs-api
logs-api: ## نمایش لاگ سرویس بک‌اند
	$(DOCKER_COMPOSE) logs -f backend

.PHONY: logs-front
logs-front: ## نمایش لاگ سرویس فرانت‌اند
	$(DOCKER_COMPOSE) logs -f frontend

.PHONY: ps
ps: ## مشاهده وضعیت سرویس‌ها
	$(DOCKER_COMPOSE) ps

.PHONY: restart
restart: ## ریستارت همه سرویس‌ها
	@echo "${YELLOW}🔄 ریستارت سرویس‌ها...${NC}"
	$(DOCKER_COMPOSE) restart
	@echo "${GREEN}✅ سرویس‌ها ریستارت شدند${NC}"

.PHONY: clean
clean: ## پاکسازی کامل (حذف کانتینرها و حجم‌ها)
	@echo "${RED}⚠ پاکسازی کامل پروژه...${NC}"
	$(DOCKER_COMPOSE) down -v
	rm -rf logs/
	rm -rf uploads/
	rm -rf pycache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "${GREEN}✅ پاکسازی انجام شد${NC}"

# ==================== Development Commands ====================
.PHONY: install
install: ## نصب وابستگی‌های پایتون
	@echo "${BLUE}📦 نصب وابستگی‌ها...${NC}"
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt
	$(PIP) install -r requirements-dev.txt 2>/dev/null || true
	@echo "${GREEN}✅ وابستگی‌ها نصب شدند${NC}"

.PHONY: dev
dev: ## اجرا در حالت توسعه (با hot-reload)
	@echo "${BLUE}🚀 اجرا در حالت توسعه...${NC}"
	$(DOCKER_COMPOSE) up --build
	.PHONY: shell
shell: ## ورود به شل بک‌اند
	$(DOCKER_COMPOSE) exec backend /bin/bash

.PHONY: shell-db
shell-db: ## ورود به شل دیتابیس
	$(DOCKER_COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

# ==================== Testing Commands ====================
.PHONY: test
test: ## اجرای تست‌ها
	@echo "${BLUE}🧪 اجرای تست‌ها...${NC}"
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-cov
test-cov: ## اجرای تست‌ها با گزارش کاور
	@echo "${BLUE}📊 اجرای تست‌ها با گزارش کاور...${NC}"
	$(PYTHON) -m pytest tests/ -v --cov=backend --cov-report=html --cov-report=term
	@echo "${GREEN}✅ گزارش کاور در htmlcov/index.html ایجاد شد${NC}"

.PHONY: test-api
test-api: ## اجرای تست‌های API
	$(PYTHON) -m pytest tests/test_api.py -v

.PHONY: test-unit
test-unit: ## اجرای تست‌های واحد
	$(PYTHON) -m pytest tests/unit/ -v

.PHONY: lint
lint: ## بررسی کیفیت کد
	@echo "${BLUE}🔍 بررسی کیفیت کد...${NC}"
	flake8 backend/ --max-line-length=120 --ignore=E203,W503
	black --check backend/
	isort --check-only backend/
	@echo "${GREEN}✅ بررسی کیفیت کد انجام شد${NC}"

.PHONY: format
format: ## فرمت کردن کد
	@echo "${BLUE}🎨 فرمت کردن کد...${NC}"
	black backend/
	isort backend/
	@echo "${GREEN}✅ کد فرمت شد${NC}"

# ==================== Model Training Commands ====================
.PHONY: download
download: ## دانلود دیتاست‌ها
	@echo "${BLUE}📥 دانلود دیتاست‌ها...${NC}"
	$(PYTHON) scripts/download_datasets.py --all

.PHONY: train
train: ## آموزش مدل YOLO
	@echo "${BLUE}🤖 آموزش مدل YOLO...${NC}"
	$(PYTHON) scripts/train_yolo.py --data dataset/ILPD --epochs 100 --batch-size 16

.PHONY: train-crnn
train-crnn: ## آموزش مدل CRNN
	@echo "${BLUE}🔤 آموزش مدل CRNN...${NC}"
	$(PYTHON) scripts/train_crnn.py --data dataset/plate_crops --epochs 50 --batch-size 32

.PHONY: train-gan
train-gan: ## آموزش مدل GAN
	@echo "${BLUE}🎨 آموزش مدل GAN...${NC}"
	$(PYTHON) scripts/train_gan.py --data dataset/ILPD --epochs 200 --batch-size 32

.PHONY: train-anomaly
train-anomaly: ## آموزش مدل تشخیص ناهنجاری
	@echo "${BLUE}🚨 آموزش مدل تشخیص ناهنجاری...${NC}"
	$(PYTHON) scripts/train_anomaly_lstm.py --epochs 100

.PHONY: export
export: ## تبدیل مدل به ONNX
	@echo "${BLUE}🔄 تبدیل مدل به ONNX...${NC}"
	$(PYTHON) scripts/export_onnx.py --all

# ==================== Database Commands ====================
.PHONY: migrate
migrate: ## اجرای مهاجرت دیتابیس
	@echo "${BLUE}🗄 اجرای مهاجرت دیتابیس...${NC}"
	$(PYTHON) backend/manage.py migrate

.PHONY: backup
backup: ## بک‌آپ گرفتن از دیتابیس
	@echo "${BLUE}💾 گرفتن بک‌آپ از دیتابیس...${NC}"
	mkdir -p backups
	docker exec anpr-postgres pg_dump -U $(POSTGRES_USER) $(POSTGRES_DB) > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "${GREEN}✅ بک‌آپ در پوشه backups ذخیره شد${NC}"

.PHONY: restore
restore: ## بازیابی دیتابیس از بک‌آپ (RESTORE_FILE=path/to/backup.sql)
	@echo "${RED}⚠ بازیابی دیتابیس از فایل $(RESTORE_FILE)${NC}"
	docker exec -i anpr-postgres psql -U $(POSTGRES_USER) $(POSTGRES_DB) < $(RESTORE_FILE)
	@echo "${GREEN}✅ دیتابیس بازیابی شد${NC}"

# ==================== Status Commands ====================
.PHONY: status
status: ## نمایش وضعیت پروژه
	@echo "${BLUE}📊 وضعیت پروژه ANPR${NC}"
	@echo "=========================================="
	@echo "${GREEN}✅ سرویس‌های در حال اجرا:${NC}"
	@$(DOCKER_COMPOSE) ps --services --filter "status=running" 2>/dev/null || echo "  هیچ سرویسی در حال اجرا نیست"
	@echo ""
	@echo "${YELLOW}📁 حجم‌های داکر:${NC}"
	@docker volume ls | grep anpr || echo "  هیچ حجمی یافت نشد"
	@echo ""
	@echo "${BLUE}🔗 آدرس‌های دسترسی:${NC}"
	@echo "  Frontend:  http://localhost:8501"
	@echo "  Backend API: http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  RabbitMQ:   http://localhost:15672 (guest/guest)"
	@echo "  MinIO:      http://localhost:9001 (minioadmin/minioadmin)"
	.PHONY: deploy
deploy: ## استقرار روی سرور (DEPLOY_HOST=user@host)
	@echo "${BLUE}🚀 استقرار روی سرور $(DEPLOY_HOST)...${NC}"
	rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '.git' --exclude 'models/*.pt' ./ $(DEPLOY_HOST):/app/anpr/
	ssh $(DEPLOY_HOST) "cd /app/anpr && docker-compose up -d --build"
	@echo "${GREEN}✅ استقرار با موفقیت انجام شد${NC}"

# ==================== Utility Commands ====================
.PHONY: info
info: ## نمایش اطلاعات پروژه
	@echo "${BLUE}📋 اطلاعات پروژه ANPR${NC}"
	@echo "=========================================="
	@echo "نام پروژه:      $(PROJECT_NAME)"
	@echo "نسخه:           3.0.0"
	@echo "پایتون:         $(shell $(PYTHON) --version 2>/dev/null || echo 'نصب نشده')"
	@echo "داکر:           $(shell docker --version 2>/dev/null || echo 'نصب نشده')"
	@echo "Docker Compose: $(shell docker-compose --version 2>/dev/null || echo 'نصب نشده')"
	@echo "پوشه پروژه:     $(shell pwd)"

.PHONY: logs-clean
logs-clean: ## پاکسازی فایل‌های لاگ
	@echo "${YELLOW}🗑 پاکسازی فایل‌های لاگ...${NC}"
	find . -type f -name "*.log" -delete
	rm -rf logs/*.log 2>/dev/null || true
	@echo "${GREEN}✅ لاگ‌ها پاکسازی شدند${NC}"

.PHONY: docker-clean
docker-clean: ## پاکسازی کامل داکر (حذف همه کانتینرها و تصاویر)
	@echo "${RED}⚠ پاکسازی کامل داکر...${NC}"
	$(DOCKER_COMPOSE) down -v
	docker system prune -af
	@echo "${GREEN}✅ داکر پاکسازی شد${NC}"

# ==================== Default target ====================
.DEFAULT_GOAL := help