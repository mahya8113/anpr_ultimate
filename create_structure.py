import os

PROJECT = "B2B"

def make_dirs_and_files(base):
    """ایجاد پوشه‌ها و فایل‌های خالی طبق درخت خواسته شده"""
    structure = {
        # فایل‌های ریشه
        ".": [
            ".env", ".gitignore", "docker-compose.yml", "Makefile", "README.md"
        ],
        # پوشه backend
        "backend": [
            "requirements.txt", "main.py"
        ],
        "backend/core": [
            "config.py", "database.py", "redis_client.py", "rabbitmq_client.py",
            "cache.py", "model_version.py", "logging_config.py", "exceptions.py"
        ],
        "backend/modules": [
            "image_formation.py", "restoration.py", "preprocessing.py",
            "feature_engineering.py", "geometry.py", "deep_core.py", "detection.py",
            "segmentation.py", "ocr_pipeline.py", "video_intelligence.py",
            "dataset_engineering.py", "model_optimization.py", "security.py",
            "multi_tracking.py", "panoptic_seg.py", "depth_estimation.py",
            "lowlight_ir.py", "domain_adaptation.py", "xai.py", "edge_ai.py",
            "advanced_ocr.py", "adversarial.py", "anomaly.py"
        ],
        "backend/api/routes": [
            "auth.py", "detect.py", "video.py", "reports.py", "license.py", "admin.py"
        ],
        "backend/api/websocket": [
            "live.py"
        ],
        "backend/api": [
            "deps.py"
        ],
        "backend/services": [
            "detection_service.py", "ocr_service.py", "tracking_service.py"
        ],
        "backend/utils": [
            "rate_limiter.py", "camera_manager.py", "metrics.py", "validators.py"
        ],
        # frontend
        "frontend": [
            "Dockerfile", "app.py"
        ],
        "frontend/pages": [
            "live.py", "upload.py", "reports.py", "settings.py", "admin.py"
        ],
        "frontend/static": [
            "style.css"
        ],
        # tests
        "tests": [
            "conftest.py", "test_api.py", "test_websocket.py",
            "test_cache.py", "test_model_version.py", "test_licensing.py"
        ],
        # scripts
        "scripts": [
            "download_datasets.py", "train_yolo.py", "train_crnn.py",
            "train_anomaly_lstm.py", "train_gan.py", "export_onnx.py",
            "migrate_model.py", "generate_project.sh"
        ],
        # models
        "models": [
            "yolov8n.pt", "yolov8n_plate_v1.pt", "crnn_persian_v1.pth", "model_metadata.json"
        ],
        # database
        "database": [
            "init.sql"
        ],
        # monitoring
        "monitoring": [
            "prometheus.yml", "grafana-dashboard.json", "alerts.yml"
        ],
        "mlflow": [],  # پوشه خالی
        # kubernetes
        "kubernetes": [
            "deployment.yaml", "service.yaml", "ingress.yaml", "hpa.yaml"
        ],
        # secrets
        "secrets": [
            "db_password.txt"
        ],
        # traefik
        "traefik": [
            "traefik.yml"
        ],
        # docs
        "docs": [
            "user_guide_fa.html"
        ]
    }

    for folder, files in structure.items():
        # مسیر کامل پوشه
        target_dir = os.path.join(base, folder) if folder != "." else base
        os.makedirs(target_dir, exist_ok=True)
        for fname in files:
            fpath = os.path.join(target_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                # می‌توانید یک خط کامنت خالی بنویسید (یا اصلاً چیزی ننویسید)
                f.write("")  # فایل خالی

def main():
    # اگر پروژه قبلاً وجود داشته باشد، هشدار می‌دهیم (اختیاری)
    root = os.path.join(os.getcwd(), PROJECT)
    if os.path.exists(root):
        resp = input(f"پوشه '{PROJECT}' از قبل وجود دارد. حذف و بازسازی؟ (y/N): ")
        if resp.lower() == 'y':
            import shutil
            shutil.rmtree(root)
        else:
            print("انجام نشد.")
            return
    make_dirs_and_files(root)
    print(f"ساختار پروژه '{PROJECT}' با موفقیت در '{root}' ایجاد شد.")

if __name__  == "__main__":
    main()