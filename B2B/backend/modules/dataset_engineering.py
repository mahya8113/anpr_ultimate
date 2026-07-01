"""
dataset_engineering.py - مهندسی دیتاست برای آموزش مدل‌های تشخیص پلاک
شامل: دانلود دیتاست‌های عمومی، تبدیل فرمت‌ها، تقسیم داده، افزایش داده (Augmentation)
"""

import os
import json
import yaml
import zipfile
import tarfile
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm
import requests
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetEngineer:
    """
    کلاس مهندسی دیتاست برای آماده‌سازی داده‌های آموزش مدل‌های ANPR
    """
    
    def init(self, base_dir: str = "dataset"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== دانلود دیتاست‌های عمومی ====================
    
    def download_ilpd(self) -> Path:
        """
        دانلود دیتاست ILPD (Iranian License Plate Dataset) از GitHub
        
        Returns:
            مسیر دیتاست دانلود شده
        """
        url = "https://github.com/mmd1426/ILPD/archive/main.zip"
        output_path = self.base_dir / "ilpd_temp.zip"
        
        logger.info("در حال دانلود دیتاست ILPD...")
        self._download_file(url, output_path)
        
        extract_path = self.base_dir / "ILPD"
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # جابجایی فایل‌ها به پوشه اصلی
        extracted_folder = extract_path / "ILPD-main"
        if extracted_folder.exists():
            for item in extracted_folder.iterdir():
                shutil.move(str(item), str(extract_path / item.name))
            extracted_folder.rmdir()
        
        output_path.unlink()
        logger.info(f"دیتاست ILPD در {extract_path} ذخیره شد")
        
        # ایجاد فایل data.yaml برای YOLO
        self._create_yolo_data_yaml(extract_path)
        
        return extract_path
    
    def download_iranis(self) -> Path:
        """
        دانلود دیتاست Iranis (حروف و اعداد فارسی)
        
        Returns:
            مسیر دیتاست دانلود شده
        """
        url = "https://github.com/niloworld/Iranis/archive/refs/heads/master.zip"
        output_path = self.base_dir / "iranis_temp.zip"
        
        logger.info("در حال دانلود دیتاست Iranis...")
        self._download_file(url, output_path)
        
        extract_path = self.base_dir / "Iranis"
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        output_path.unlink()
        logger.info(f"دیتاست Iranis در {extract_path} ذخیره شد")
        return extract_path
    
    def download_kaggle_plate(self, dataset_name: str = "iranian-license-plates") -> Path:
        """
        دانلود دیتاست از Kaggle (نیاز به Kaggle API)
        
        Args:
            dataset_name: نام دیتاست در Kaggle
        
        Returns:
            مسیر دیتاست
        """
        try:
            import kaggle
            output_path = self.base_dir / "kaggle_plates"
            kaggle.api.dataset_download_files(dataset_name, path=output_path, unzip=True)
            logger.info(f"دیتاست Kaggle در {output_path} ذخیره شد")
            return output_path
        except ImportError:
            logger.error("Kaggle API نصب نیست. نصب: pip install kaggle")
            raise
    
    def _download_file(self, url: str, dest: Path):
        """دانلود فایل با نمایش پیشرفت"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # ==================== تبدیل فرمت‌ها ====================
    
    def _create_yolo_data_yaml(self, dataset_path: Path):
        """
        ایجاد فایل data.yaml برای آموزش YOLO
        """
        data_config = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['license_plate']
        }
        
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"فایل data.yaml در {yaml_path} ایجاد شد")
    
    def convert_coco_to_yolo(self, coco_json_path: Path, output_dir: Path, img_dir: Path):
        """
        تبدیل آنوتیشن COCO (JSON) به فرمت YOLO (.txt)
        
        Args:
            coco_json_path: مسیر فایل JSON آنوتیشن COCO
            output_dir: پوشه خروجی برای فایل‌های YOLO
            img_dir: پوشه تصاویر (برای تطابق عرض و ارتفاع)
        """
        import json
        
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
        
        # ایجاد نگاشت از image_id به اطلاعات تصویر
        images_info = {img['id']: img for img in coco['images']}
        categories = {cat['id']: cat for cat in coco['categories']}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # گروه‌بندی آنوتیشن‌ها بر اساس image_id
        anns_by_image = {}
        for ann in coco['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)
        
        for img_id, anns in anns_by_image.items():
            img_info = images_info[img_id]
            img_w = img_info['width']
            img_h = img_info['height']
            
            # نام فایل تصویر
            img_filename = Path(img_info['file_name']).stem
            txt_path = output_dir / f"{img_filename}.txt"
            
            with open(txt_path, 'w') as f:
                for ann in anns:
                    # COCO bbox: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    # تبدیل به مختصات مرکزی نرمال‌شده YOLO
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    width_norm = w / img_w
                    height_norm = h / img_h
                    
                    class_id = ann['category_id'] - 1  # فرض می‌کنیم کلاس‌ها از 1 شروع شوند
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        logger.info(f"تبدیل COCO به YOLO انجام شد: {len(anns_by_image)} فایل ایجاد شد")
    
    def convert_voc_to_yolo(self, voc_xml_dir: Path, output_dir: Path):
        """
        تبدیل آنوتیشن PASCAL VOC (XML) به فرمت YOLO (.txt)
        
        Args:
            voc_xml_dir: پوشه حاوی فایل‌های XML
            output_dir: پوشه خروجی برای فایل‌های YOLO
        """
        import xml.etree.ElementTree as ET
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for xml_path in voc_xml_dir.glob("*.xml"):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # اندازه تصویر
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
            txt_path = output_dir / f"{xml_path.stem}.txt"
            
            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    # فرض می‌کنیم کلاس‌ها: 'license_plate' => 0
                    class_id = 0 if class_name.lower() in ['license_plate', 'plate'] else -1
                    if class_id == -1:
                        continue
                    
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    # تبدیل به YOLO
                    x_center = (xmin + xmax) / 2 / img_w
                    y_center = (ymin + ymax) / 2 / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"تبدیل VOC به YOLO انجام شد: {len(list(voc_xml_dir.glob('*.xml')))} فایل ایجاد شد")
    
    # ==================== تقسیم داده ====================
    
    def split_dataset(
        self,
        images_dir: Path,
        labels_dir: Optional[Path],
        output_dir: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        تقسیم دیتاست به train, val, test
        
        Args:
            images_dir: پوشه تصاویر
            labels_dir: پوشه لیبل‌ها (اختیاری)
            output_dir: پوشه خروجی با ساختار YOLO
            train_ratio, val_ratio, test_ratio: نسبت‌ها (مجموعاً 1)
            seed: seed برای reproducibility
        """
        random.seed(seed)
        
        # ایجاد ساختار پوشه‌ها
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            if labels_dir:
                (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # دریافت لیست فایل‌ها
        image_files = list(images_dir.glob("*.[jJ][pP][gG]")) + \
                      list(images_dir.glob("*.[jJ][pP][eE][gG]")) + \
                      list(images_dir.glob("*.[pP][nN][gG]"))
        
        # تقسیم
        train_files, temp_files = train_test_split(image_files, test_size=(1 - train_ratio), random_state=seed)
        val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            for img_path in files:
                # کپی تصویر
                dest_img = output_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # کپی لیبل (در صورت وجود)
                if labels_dir:
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        dest_label = output_dir / split_name / 'labels' / f"{img_path.stem}.txt"
                        shutil.copy2(label_path, dest_label)
        
        logger.info(f"تقسیم داده انجام شد: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
        
        # ایجاد فایل data.yaml
        self._create_yolo_data_yaml(output_dir)
    
    # ==================== افزایش داده (Augmentation) ====================
    
    @staticmethod
    def augment_image(image: np.ndarray, labels: List[Tuple[int, float, float, float, float]]) -> Tuple[np.ndarray, List]:
        """
        اعمال augmentations روی تصویر و لیبل‌های متناظر
        
        Args:
            image: تصویر BGR
            labels: لیست لیبل‌های YOLO [class_id, x_center, y_center, width, height]
            Returns:
            (image_augmented, labels_augmented)
        """
        import albumentations as A
        
        # تبدیل لیبل‌ها به فرمت albumentations
        bboxes = []
        for label in labels:
            class_id, x_c, y_c, w, h = label
            # تبدیل از مرکزی به x_min, y_min, x_max, y_max
            x_min = (x_c - w/2) * image.shape[1]
            y_min = (y_c - h/2) * image.shape[0]
            x_max = (x_c + w/2) * image.shape[1]
            y_max = (y_c + h/2) * image.shape[0]
            bboxes.append([x_min, y_min, x_max, y_max, class_id])
        
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        class_labels = [b[4] for b in bboxes]
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        # تبدیل برگشتی به فرمت YOLO
        new_labels = []
        for bbox in transformed['bboxes']:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / transformed['image'].shape[1]
            y_center = (y_min + y_max) / 2 / transformed['image'].shape[0]
            width = (x_max - x_min) / transformed['image'].shape[1]
            height = (y_max - y_min) / transformed['image'].shape[0]
            new_labels.append([bbox[4], x_center, y_center, width, height])
        
        return transformed['image'], new_labels
    
    def create_augmented_dataset(self, input_dir: Path, output_dir: Path, num_augmented: int = 5):
        """
        ایجاد دیتاست افزایش یافته با کپی و augment تصاویر
        
        Args:
            input_dir: پوشه ورودی با ساختار YOLO (images, labels)
            output_dir: پوشه خروجی
            num_augmented: تعداد تصاویر جدید برای هر تصویر اصلی
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = input_dir / 'images'
        labels_dir = input_dir / 'labels'
        
        if not images_dir.exists():
            logger.error("پوشه images یافت نشد")
            return
        
        output_images = output_dir / 'images'
        output_labels = output_dir / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # کپی تصاویر اصلی
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_path, output_images / img_path.name)
                
                # کپی لیبل متناظر
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, output_labels / f"{img_path.stem}.txt")
        
        # تولید تصاویر augment شده
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # خواندن لیبل‌ها
            label_path = labels_dir / f"{img_path.stem}.txt"
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                            if not labels:
                             continue
            
            for i in range(num_augmented):
                aug_img, aug_labels = self.augment_image(image, labels)
                new_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(output_images / new_name), aug_img)
                
                # ذخیره لیبل‌های جدید
                with open(output_labels / f"{Path(new_name).stem}.txt", 'w') as f:
                    for lbl in aug_labels:
                        f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        
        logger.info(f"دیتاست افزایش یافته در {output_dir} با {num_augmented} برابر داده جدید ایجاد شد")
    
    # ==================== ایجاد دیتاست CRNN برای OCR ====================
    
    def create_crnn_dataset(self, images_dir: Path, labels_file: Path, output_dir: Path):
        """
        آماده‌سازی دیتاست برای آموزش CRNN (تشخیص حروف پلاک)
        
        Args:
            images_dir: پوشه تصاویر برش خورده پلاک
            labels_file: فایل JSON با نگاشت filename -> text
            output_dir: پوشه خروجی با ساختار train/val
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        # لیست فایل‌ها
        all_files = list(images_dir.glob("*.[jJ][pP][gG]")) + \
                    list(images_dir.glob("*.[jJ][pP][eE][gG]")) + \
                    list(images_dir.glob("*.[pP][nN][gG]"))
        
        # تقسیم
        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
        
        # ایجاد پوشه‌ها
        for split in ['train', 'val']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        
        # ذخیره لیبل‌ها به صورت JSON در هر پوشه
        def save_split(files, split_name):
            split_labels = {}
            for img_path in files:
                text = labels.get(img_path.name, "")
                if not text:
                    continue
                dest_img = output_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                split_labels[img_path.name] = text
            
            with open(output_dir / split_name / 'labels.json', 'w', encoding='utf-8') as f:
                json.dump(split_labels, f, ensure_ascii=False, indent=2)
        
        save_split(train_files, 'train')
        save_split(val_files, 'val')
        
        logger.info(f"دیتاست CRNN ایجاد شد: {len(train_files)} train, {len(val_files)} val")
    
    # ==================== ابزارهای کمکی ====================
    
    def validate_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """
        اعتبارسنجی دیتاست YOLO (بررسی وجود تصاویر و لیبل‌ها)
        
        Returns:
            دیکشنری شامل آمار و خطاها
        """
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        
        if not images_dir.exists():
            return {"error": "پوشه images یافت نشد"}
        
        image_files = list(images_dir.glob("*.*"))
        label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        
        missing_labels = []
        for img in image_files:
            label_path = labels_dir / f"{img.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img.name)
        
        stats = {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "missing_labels": len(missing_labels),
            "missing_list": missing_labels[:20],  # فقط 20 عدد اول
            "valid": len(missing_labels) == 0
        }
        
        logger.info(f"آمار دیتاست: {stats['total_images']} تصویر, {stats['total_labels']} لیبل, {stats['missing_labels']}缺失")
        return stats