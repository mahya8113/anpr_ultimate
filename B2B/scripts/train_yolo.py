#!/usr/bin/env python3
"""
train_yolo.py
آموزش مدل YOLOv8 برای تشخیص پلاک خودروهای ایرانی
پشتیبانی از: آموزش از صفر، فاین‌تون، ارزیابی، تبدیل به ONNX
"""

import os
import sys
import yaml
import torch
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==================== تنظیمات ====================
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    if status == "success":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠️ {message}{RESET}")
    elif status == "error":
        print(f"{RED}❌ {message}{RESET}")
    else:
        print(f"{BLUE}📌 {message}{RESET}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print_status(f"دستگاه مورد استفاده: {DEVICE}", "info")


# ==================== ایجاد فایل data.yaml ====================
def create_data_yaml(data_dir, output_path='dataset.yaml'):
    """
    ایجاد فایل data.yaml برای YOLO از ساختار دیتاست
    ساختار مورد نیاز:
        dataset/
            train/
                images/
                labels/
            val/
                images/
                labels/
    """
    data_config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # تعداد کلاس‌ها (فقط پلاک)
        'names': ['license_plate'],
        'channels': 3
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
    
    print_status(f"فایل data.yaml در {output_path} ایجاد شد", "success")
    return output_path


# ==================== آموزش مدل ====================
def train_yolo(
    data_yaml,
    model_name='yolov8n.pt',
    epochs=100,
    imgsz=640,
    batch_size=16,
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    workers=8,
    augment=True,
    patience=20,
    save_period=10,
    project='models',
    name='yolov8n_plate',
    exist_ok=True,
    resume=False
):
    """
    آموزش مدل YOLOv8 برای تشخیص پلاک
    """
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print(f"🚀 شروع آموزش مدل {model_name}")
    print("="*60)
    print_status(f"دیتاست: {data_yaml}", "info")
    print_status(f"تعداد دوره‌ها: {epochs}", "info")
    print_status(f"سایز تصاویر: {imgsz}", "info")
    print_status(f"سایز دسته: {batch_size}", "info")
    print_status(f"نرخ یادگیری: {lr}", "info")
    
    # بارگذاری مدل
    model = YOLO(model_name)
    
    # آموزش
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        workers=workers,
        device=DEVICE,
        augment=augment,
        patience=patience,
        save_period=save_period,
        project=project,
        name=name,
        exist_ok=exist_ok,
        resume=resume,
        seed=42,
        deterministic=True,
        single_cls=True,  # فقط یک کلاس (پلاک)
        verbose=True
    )
    
    # مسیر مدل نهایی
    model_path = Path(project) / name / 'weights' / 'best.pt'
    print_status(f"آموزش به پایان رسید!", "success")
    print_status(f"بهترین مدل در {model_path} ذخیره شد", "success")
    
    return results, model_path


# ==================== اعتبارسنجی ====================
def validate_model(model_path, data_yaml):
    """
    اعتبارسنجی مدل روی دیتاست validation
    """
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("🔍 اعتبارسنجی مدل")
    print("="*60)
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    print_status(f"mAP50: {metrics.box.map50:.4f}", "success")
    print_status(f"mAP50-95: {metrics.box.map:.4f}", "success")
    print_status(f"Precision: {metrics.box.mp:.4f}", "success")
    print_status(f"Recall: {metrics.box.mr:.4f}", "success")
    
    return metrics


# ==================== تست روی تصاویر ====================
def test_model(model_path, test_images_dir, output_dir='test_results'):
    """
    تست مدل روی تصاویر نمونه و ذخیره نتایج
    """
    from ultralytics import YOLO
    
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    
    # دریافت لیست تصاویر
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(Path(test_images_dir).glob(ext)))
    
    print_status(f"تست روی {len(test_images)} تصویر", "info")
    
    results_list = []
    for img_path in test_images:
        results = model(img_path)
        
        # ذخیره تصویر با آنوتیشن
        output_path = os.path.join(output_dir, f"result_{img_path.name}")
        results[0].save(output_path)
        
        # استخراج نتایج
        boxes = results[0].boxes
        if boxes is not None:
            num_plates = len(boxes)
            confs = boxes.conf.cpu().numpy()
            results_list.append({
                'image': img_path.name,
                'num_plates': num_plates,
                'max_confidence': max(confs) if len(confs) > 0 else 0
            })
    
    # نمایش خلاصه
    df = pd.DataFrame(results_list)
    print(df.to_string())
    
    return results_list


# ==================== رسم نمودارهای آموزش ====================
def plot_training_results(results_dir):
    """
    رسم نمودارهای loss، precision، recall و mAP
    """
    results_file = Path(results_dir) / 'results.csv'
    
    if not results_file.exists():
        print_status(f"فایل {results_file} یافت نشد", "warning")
        return
    
    df = pd.read_csv(results_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Precision/Recall
    axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision and Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # mAP
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('Mean Average Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'training_plots.png')
    plt.close()
    
    print_status(f"نمودارها در {results_dir}/training_plots.png ذخیره شد", "success")


# ==================== تبدیل به ONNX ====================
def export_to_onnx(model_path, imgsz=640):
    """
    تبدیل مدل YOLO به فرمت ONNX
    """
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("🔄 تبدیل مدل به ONNX")
    print("="*60)
    
    model = YOLO(model_path)
    onnx_path = model.export(format='onnx', imgsz=imgsz, half=True)
    
    print_status(f"مدل ONNX در {onnx_path} ذخیره شد", "success")
    return onnx_path
# ==================== فاین‌تون روی دیتاست اختصاصی ====================
def fine_tune(model_path, data_yaml, epochs=50):
    """
    فاین‌تون مدل روی دیتاست اختصاصی
    """
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("🔧 فاین‌تون مدل روی دیتاست اختصاصی")
    print("="*60)
    
    model = YOLO(model_path)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        lr0=0.001,  # نرخ یادگیری کمتر برای فاین‌تون
        device=DEVICE,
        name='fine_tuned_plate',
        exist_ok=True
    )
    
    print_status("فاین‌تون با موفقیت انجام شد!", "success")
    return results


# ==================== تابع اصلی ====================
def main():
    parser = argparse.ArgumentParser(description='آموزش مدل YOLO برای تشخیص پلاک خودرو')
    parser.add_argument('--data', type=str, default='dataset/ILPD', help='مسیر دیتاست')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='مدل پایه (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--epochs', type=int, default=100, help='تعداد دوره‌های آموزش')
    parser.add_argument('--batch-size', type=int, default=16, help='سایز دسته')
    parser.add_argument('--imgsz', type=int, default=640, help='سایز تصاویر ورودی')
    parser.add_argument('--lr', type=float, default=0.01, help='نرخ یادگیری اولیه')
    parser.add_argument('--name', type=str, default='yolov8n_plate', help='نام پروژه')
    parser.add_argument('--resume', action='store_true', help='ادامه آموزش از آخرین checkpoint')
    parser.add_argument('--validate', action='store_true', help='فقط اعتبارسنجی مدل موجود')
    parser.add_argument('--test', type=str, help='تست مدل روی پوشه تصاویر')
    parser.add_argument('--onnx', action='store_true', help='تبدیل به ONNX پس از آموزش')
    parser.add_argument('--fine-tune', type=str, help='فاین‌تون مدل روی دیتاست جدید')
    
    args = parser.parse_args()
    
    # فقط اعتبارسنجی
    if args.validate:
        if not os.path.exists(args.model):
            print_status(f"مدل {args.model} یافت نشد", "error")
            sys.exit(1)
        data_yaml = create_data_yaml(args.data)
        validate_model(args.model, data_yaml)
        return
    
    # فقط تست
    if args.test:
        if not os.path.exists(args.model):
            print_status(f"مدل {args.model} یافت نشد", "error")
            sys.exit(1)
        test_model(args.model, args.test)
        return
    
    # فاین‌تون
    if args.fine_tune:
        if not os.path.exists(args.model):
            print_status(f"مدل {args.model} یافت نشد", "error")
            sys.exit(1)
        data_yaml = create_data_yaml(args.fine_tune)
        fine_tune(args.model, data_yaml, args.epochs)
        return
    
    # آموزش از ابتدا
    data_yaml = create_data_yaml(args.data)
    
    results, model_path = train_yolo(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        lr=args.lr,
        name=args.name,
        resume=args.resume
    )
    
    # اعتبارسنجی مدل نهایی
    validate_model(model_path, data_yaml)
    
    # رسم نمودارها
    plot_training_results(f'models/{args.name}')
    
    # تبدیل به ONNX
    if args.onnx:
        export_to_onnx(model_path, args.imgsz)
    
    print("\n" + "="*60)
    print_status("آموزش با موفقیت به پایان رسید!", "success")
    print("="*60)


if __name__ == "__main__":
    main()