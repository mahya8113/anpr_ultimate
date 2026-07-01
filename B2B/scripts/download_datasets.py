#!/usr/bin/env python3
"""
download_datasets.py
اسکریپت دانلود خودکار تمام دیتاست‌های مورد نیاز برای آموزش مدل‌های تشخیص پلاک

دیتاست‌های دانلودی:
1. ILPD (Iranian License Plate Dataset) - دیتاست پلاک ایران
2. Iranis - دیتاست حروف و اعداد فارسی
3. Synthetic Plates (اختیاری) - پلاک‌های مصنوعی
"""

import os
import sys
import zipfile
import tarfile
import requests
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import hashlib

# ==================== تنظیمات ====================
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    """چاپ پیام با رنگ"""
    if status == "success":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠️ {message}{RESET}")
    elif status == "error":
        print(f"{RED}❌ {message}{RESET}")
    elif status == "info":
        print(f"{BLUE}📌 {message}{RESET}")
    else:
        print(f"📌 {message}")

def create_directories():
    """ایجاد پوشه‌های مورد نیاز"""
    dirs = [
        'dataset',
        'dataset/ILPD',
        'dataset/Iranis',
        'dataset/synthetic',
        'dataset/raw_images',
        'dataset/annotations'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print_status("پوشه‌های دیتاست ایجاد شدند", "success")

def download_file(url, dest_path, chunk_size=8192):
    """دانلود فایل با نمایش پیشرفت"""
    try:
        # بررسی وجود فایل قبلی
        if os.path.exists(dest_path):
            print_status(f"{dest_path} از قبل وجود دارد، در حال بررسی کامل بودن...", "warning")
            # می‌توانید بررسی کنید که فایل کامل است یا خیر
            return True
        
        print_status(f"در حال دانلود از {url}", "info")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print_status(f"دانلود شد: {dest_path}", "success")
        return True
        
    except Exception as e:
        print_status(f"خطا در دانلود {url}: {e}", "error")
        return False

def extract_zip(zip_path, extract_to):
    """استخراج فایل zip"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print_status(f"استخراج شد: {zip_path} -> {extract_to}", "success")
        return True
    except Exception as e:
        print_status(f"خطا در استخراج {zip_path}: {e}", "error")
        return False

def extract_tar(tar_path, extract_to):
    """استخراج فایل tar.gz"""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        print_status(f"استخراج شد: {tar_path} -> {extract_to}", "success")
        return True
    except Exception as e:
        print_status(f"خطا در استخراج {tar_path}: {e}", "error")
        return False

# ==================== دانلود دیتاست ILPD ====================
def download_ilpd():
    """دانلود دیتاست ILPD (Iranian License Plate Dataset)"""
    print("\n" + "="*50)
    print("📦 1. دانلود دیتاست ILPD (پلاک ایران)")
    print("="*50)
    
    ilpd_dir = "dataset/ILPD"
    zip_path = "dataset/ilpd_temp.zip"
    
    # لینک‌های مختلف برای دانلود (در صورت عدم دسترسی به یکی، دیگری استفاده شود)
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/ILPD.zip",
        "https://www.kaggle.com/api/v1/datasets/dataturk/ilpd/download",
        "https://github.com/mmd1426/ILPD/archive/refs/heads/main.zip"
    ]
    
    for url in urls:
        print_status(f"تلاش برای دانلود از: {url}")
        if download_file(url, zip_path):
            break
        else:
            print_status("تلاش ناموفق، امتحان لینک بعدی...", "warning")
            continue
    else:
        print_status("دانلود ILPD از همه لینک‌ها ناموفق بود", "error")
        print_status("لطفاً به صورت دستی از https://github.com/mmd1426/ILPD دانلود کنید", "warning")
        return False
    
    # استخراج
    extract_zip(zip_path, ilpd_dir)
    
    # حذف فایل موقت
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # ایجاد فایل data.yaml برای YOLO
    create_yolo_data_yaml(ilpd_dir)
    
    return True

def create_yolo_data_yaml(data_dir):
    """ایجاد فایل data.yaml برای آموزش YOLO"""
    data_yaml = {
        'path': os.path.abspath(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['license_plate'],
        'channels': 3
    }
    
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print_status(f"فایل data.yaml در {yaml_path} ایجاد شد", "success")
    return yaml_path

# ==================== دانلود دیتاست Iranis (حروف و اعداد فارسی) ====================
def download_iranis():
    """دانلود دیتاست Iranis - حروف و اعداد فارسی"""
    print("\n" + "="*50)
    print("📦 2. دانلود دیتاست Iranis (حروف و اعداد فارسی)")
    print("="*50)
    
    iranis_dir = "dataset/Iranis"
    zip_path = "dataset/iranis_temp.zip"
    
    # لینک دانلود (در صورت وجود)
    urls = [
        "https://github.com/niloworld/Iranis/releases/download/v1.0/Iranis.zip",
    ]
    
    for url in urls:
        if download_file(url, zip_path):
            break
    
    if os.path.exists(zip_path):
        extract_zip(zip_path, iranis_dir)
        os.remove(zip_path)
        create_iranis_metadata(iranis_dir)
        return True
    else:
        print_status("دانلود Iranis ناموفق بود", "error")
        print_status("می‌توانید از EasyOCR استفاده کنید یا دیتاست را از https://github.com/niloworld/Iranis دانلود کنید", "warning")
        return False

def create_iranis_metadata(data_dir):
    """ایجاد فایل متادیتا برای دیتاست Iranis"""
    metadata = {
        "name": "Iranis",
        "description": "دیتاست حروف و اعداد فارسی برای OCR",
        "classes": ["alef", "be", "pe", "te", "se", ...],
        "total_images": 83000,
        "format": "images",
        "usage": "character_recognition"
    }
    
    meta_path = os.path.join(data_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print_status(f"متادیتا در {meta_path} ایجاد شد", "success")

# ==================== دانلود دیتاست مصنوعی (اختیاری) ====================
def download_synthetic_dataset(num_samples=1000):
    """تولید دیتاست مصنوعی پلاک"""
    print("\n" + "="*50)
    print("📦 3. تولید دیتاست مصنوعی پلاک (اختیاری)")
    print("="*50)
    
    synthetic_dir = "dataset/synthetic"
    
    # کد تولید دیتاست مصنوعی ساده
    print_status(f"در حال تولید {num_samples} تصویر مصنوعی پلاک...", "info")
    
    try:
        # استفاده از کتابخانه synthetic-plate-generator اگر نصب باشد
        import random
        from PIL import Image, ImageDraw, ImageFont
        
        for i in tqdm(range(num_samples), desc="تولید تصاویر مصنوعی"):
            # ایجاد تصویر خالی
            img = Image.new('RGB', (320, 100), color='white')
            draw = ImageDraw.Draw(img)
            
            # تولید پلاک تصادفی
            plate = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.choice(['ب','س','ط','ص','د'])}{random.randint(10,99)}"
            # ذخیره تصویر (ساده شده)
            img.save(f"{synthetic_dir}/plate_{i:06d}.jpg")
            
            # ذخیره برچسب
            with open(f"{synthetic_dir}/labels_{i:06d}.txt", 'w') as f:
                f.write(plate)
        
        print_status(f"{num_samples} تصویر مصنوعی در {synthetic_dir} تولید شد", "success")
        return True
        
    except Exception as e:
        print_status(f"خطا در تولید دیتاست مصنوعی: {e}", "error")
        return False

# ==================== ایجاد ساختار YOLO برای ILPD ====================
def create_yolo_structure():
    """ایجاد ساختار پوشه‌های YOLO برای ILPD"""
    base_dir = "dataset/ILPD"
    
    # ایجاد پوشه‌های YOLO
    for split in ['train', 'val', 'test']:
        Path(f"{base_dir}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{base_dir}/{split}/labels").mkdir(parents=True, exist_ok=True)
    
    print_status("ساختار پوشه‌های YOLO ایجاد شد", "success")

# ==================== نصب وابستگی‌ها ====================
def install_dependencies():
    """نصب کتابخانه‌های مورد نیاز"""
    print("\n" + "="*50)
    print("📦 نصب وابستگی‌های مورد نیاز")
    print("="*50)
    
    dependencies = [
        'requests',
        'tqdm',
        'pyyaml',
        'pillow',
        'opencv-python-headless',
        'ultralytics'
    ]
    
    for dep in dependencies:
        print_status(f"نصب {dep}...", "info")
        try:
            __import__ (dep.replace('-', '_'))
            print_status(f"{dep} از قبل نصب است", "success")
        except ImportError:
            os.system(f"pip install {dep} -q")
            print_status(f"{dep} نصب شد", "success")
    
    # نصب easyocr برای OCR
    try:
        import easyocr
        print_status("easyocr از قبل نصب است", "success")
    except ImportError:
        os.system("pip install easyocr -q")
        print_status("easyocr نصب شد", "success")

# ==================== تابع اصلی ====================
def main():
    parser = argparse.ArgumentParser(description='دانلود دیتاست‌های مورد نیاز پروژه ANPR')
    parser.add_argument('--ilpd', action='store_true', help='دانلود دیتاست ILPD')
    parser.add_argument('--iranis', action='store_true', help='دانلود دیتاست Iranis')
    parser.add_argument('--synthetic', type=int, nargs='?', const=1000, help='تولید دیتاست مصنوعی (تعداد نمونه)')
    parser.add_argument('--all', action='store_true', help='دانلود همه دیتاست‌ها')
    parser.add_argument('--no-deps', action='store_true', help='عدم نصب وابستگی‌ها')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚗 دانلود خودکار دیتاست‌های سامانه تشخیص پلاک")
    print("="*60)
    
    # نصب وابستگی‌ها
    if not args.no_deps:
        install_dependencies()
    
    # ایجاد پوشه‌ها
    create_directories()
    create_yolo_structure()
    
    # دانلود دیتاست‌ها
    if args.all or args.ilpd:
        download_ilpd()
    
    if args.all or args.iranis:
        download_iranis()
    
    if args.all or args.synthetic:
        num = args.synthetic if args.synthetic else 1000
        download_synthetic_dataset(num)
    
    # اگر هیچ گزینه‌ای داده نشده، راهنما را نمایش بده
    if not (args.ilpd or args.iranis or args.synthetic or args.all):
        print_status("هیچ گزینه‌ای انتخاب نشده است", "warning")
        print("""
        روش استفاده:
        ------------
        python download_datasets.py --ilpd          # فقط دانلود ILPD
        python download_datasets.py --iranis        # فقط دانلود Iranis
        python download_datasets.py --synthetic 500 # تولید 500 تصویر مصنوعی
        python download_datasets.py --all           # دانلود همه
        
        راهنمای بیشتر:
        python download_datasets.py --help
        """)
    
    print("\n" + "="*60)
    print_status("عملیات دانلود تکمیل شد!", "success")
    print("="*60)

if __name__ == "__main__":
    main()