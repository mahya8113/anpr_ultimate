#!/usr/bin/env python3
"""
train_crnn.py
آموزش مدل CRNN (Convolutional Recurrent Neural Network) برای تشخیص حروف و اعداد فارسی پلاک خودرو
پشتیبانی از: حروف فارسی، اعداد فارسی، اعداد انگلیسی
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import argparse
from datetime import datetime
import random
from tqdm import tqdm

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_status(f"دستگاه مورد استفاده: {DEVICE}", "info")

# ==================== کاراکترهای مجاز ====================
# حروف فارسی مجاز در پلاک
PERSIAN_CHARS = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
# اعداد فارسی
PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹'
# اعداد انگلیسی
ENGLISH_DIGITS = '0123456789'

# ترکیب همه کاراکترها
ALL_CHARS = PERSIAN_CHARS + PERSIAN_DIGITS + ENGLISH_DIGITS
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(ALL_CHARS)}  # 0 برای blank
IDX2CHAR = {i + 1: ch for i, ch in enumerate(ALL_CHARS)}
NUM_CLASSES = len(ALL_CHARS) + 1  # +1 برای blank

print_status(f"تعداد کلاس‌ها: {NUM_CLASSES} (شامل blank)", "info")
print_status(f"کاراکترهای پشتیبانی شده: {ALL_CHARS}", "info")


# ==================== مدل CRNN ====================
class CRNN(nn.Module):
    """
    مدل CRNN برای تشخیص متن از تصویر
    شامل: CNN برای استخراج ویژگی، RNN (LSTM) برای توالی، CTC برای Decoding
    """
    def init(self, num_classes, hidden_size=256, input_height=48, input_width=160):
        super(CRNN, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # ========== CNN بخش ==========
        self.cnn = nn.Sequential(
            # لایه 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 48 -> 24
            
            # لایه 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 24 -> 12
            
            # لایه 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 12 -> 6
            
            # لایه 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),  # 6 -> 3
        )
        
        # محاسبه اندازه خروجی CNN
        self.cnn_output_height = input_height // 16  # 48/16 = 3
        self.cnn_output_channels = 512
        
        # ========== RNN بخش ==========
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_channels * self.cnn_output_height,  # 512 * 3 = 1536
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        
        # ========== Fully Connected ==========
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, channel, height, width)
        batch_size = x.size(0)
        
        # CNN
        features = self.cnn(x)  # (batch, 512, 3, width/4)
        
        # آماده‌سازی برای RNN: (batch, width/4, 512 * 3)
        features = features.permute(0, 3, 1, 2)  # (batch, width, channel, height)
        features = features.reshape(batch_size, -1, features.size(1) * features.size(2))
        
        # RNN
        rnn_out, _ = self.rnn(features)  # (batch, width, hidden*2)
        
        # FC
        output = self.fc(rnn_out)  # (batch, width, num_classes)
        
        return output


# ==================== دیتاست ====================
class PersianPlateDataset(Dataset):
    """
    دیتاست تصاویر پلاک با برچسب‌های متنی
    ساختار پوشه:
        dataset/
            images/
                plate_001.jpg
                plate_002.jpg
            labels.json
    یا
        dataset/
            train/
                images/
                labels.json
            val/
    """
    def init(self, data_dir, img_height=48, img_width=160, augment=False, is_train=True):
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment and is_train
        self.is_train = is_train
        
        # تعیین مسیر تصاویر و لیبل‌ها
        if is_train:
            self.image_dir = self.data_dir / 'train' / 'images'
            label_file = self.data_dir / 'train' / 'labels.json'
        else:
            self.image_dir = self.data_dir / 'val' / 'images'
            label_file = self.data_dir / 'val' / 'labels.json'
        
        # اگر ساختار ساده باشد
        if not self.image_dir.exists():
            self.image_dir = self.data_dir / 'images'
            label_file = self.data_dir / 'labels.json'
        
        if not self.image_dir.exists():
            raise ValueError(f"پوشه تصاویر در {self.image_dir} یافت نشد")
        
        # بارگذاری برچسب‌ها
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        self.images = [f for f in self.image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        self.images = [f for f in self.images if f.name in self.labels]
        
        print_status(f"دیتاست {'آموزش' if is_train else 'اعتبارسنجی'}: {len(self.images)} تصویر", "info")
    
    def len(self):
        return len(self.images)
    
    def getitem(self, idx):
        img_path = self.images[idx]
        text = self.labels[img_path.name]
        
        # بارگذاری تصویر
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"تصویر {img_path} یافت نشد")
        
        # تغییر اندازه
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # اعمال Augmentation
        if self.augment:
            img = self._apply_augmentation(img)
        
        # نرمال‌سازی
        img = img.astype(np.float32) / 255.0
        
        # تبدیل به تانسور
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        
        # تبدیل متن به ایندکس
        target = []
        for ch in text:
            if ch in CHAR2IDX:
                target.append(CHAR2IDX[ch])
        target = torch.tensor(target)
        
        return img, target, text
    
    def _apply_augmentation(self, img):
        """اعمال Augmentation روی تصویر"""
        # چرخش تصادفی
        if random.random() > 0.7:
            angle = random.uniform(-5, 5)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        
        # تغییر روشنایی تصادفی
        if random.random() > 0.7:
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
        # اضافه کردن نویز
        if random.random() > 0.8:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            return img
    
    @staticmethod
    def collate_fn(batch):
        """ترکیب batch با طول‌های مختلف"""
        images, targets, texts = zip(*batch)
        images = torch.stack(images)
        
        # محاسبه طول هدف‌ها
        target_lengths = torch.tensor([len(t) for t in targets])
        targets_concat = torch.cat(targets)
        
        return images, targets_concat, target_lengths, texts


# ==================== توابع کمکی ====================
def decode_predictions(preds):
    """تبدیل خروجی مدل به متن"""
    preds = preds.permute(1, 0, 2)  # (T, B, C)
    preds = torch.softmax(preds, dim=2)
    preds = torch.argmax(preds, dim=2)  # (T, B)
    
    batch_size = preds.size(1)
    texts = []
    
    for b in range(batch_size):
        text = ''
        prev = 0
        for t in range(preds.size(0)):
            char_idx = preds[t, b].item()
            if char_idx != 0 and char_idx != prev:
                text += IDX2CHAR.get(char_idx, '')
            prev = char_idx
        texts.append(text)
    
    return texts


# ==================== آموزش مدل ====================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """حلقه آموزش مدل CRNN"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # ========== آموزش ==========
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, targets, target_lengths, texts in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # پیش‌بینی
            logits = model(images)  # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            
            # محاسبه طول ورودی
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            
            # محاسبه loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # محاسبه دقت
            with torch.no_grad():
                preds = decode_predictions(logits.cpu())
                for pred, true in zip(preds, texts):
                    if pred == true:
                        train_correct += 1
                    train_total += 1
            
            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total if train_total > 0 else 0})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # ========== اعتبارسنجی ==========
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets, target_lengths, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                
                logits = model(images)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                
                input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
                val_loss += loss.item()
                
                preds = decode_predictions(logits.cpu())
                for pred, true in zip(preds, texts):
                    if pred == true:
                        val_correct += 1
                    val_total += 1
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # ذخیره بهترین مدل
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/crnn_persian_best.pth')
            print_status(f"Best model saved with loss: {best_val_loss:.4f}", "success")
    
    return train_losses, val_losses, train_accs, val_accs


# ==================== تابع اصلی ====================
def main():
    parser = argparse.ArgumentParser(description='آموزش مدل CRNN برای تشخیص حروف و اعداد فارسی پلاک')
    parser.add_argument('--data', type=str, default='dataset/plate_crops', help='مسیر دیتاست')
    parser.add_argument('--epochs', type=int, default=50, help='تعداد دوره‌های آموزش')
    parser.add_argument('--batch-size', type=int, default=32, help='سایز دسته')
    parser.add_argument('--lr', type=float, default=0.001, help='نرخ یادگیری')
    parser.add_argument('--hidden-size', type=int, default=256, help='سایز لایه پنهان LSTM')
    parser.add_argument('--save', type=str, default='models/crnn_persian_v1.pth', help='مسیر ذخیره مدل')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔤 آموزش مدل CRNN برای تشخیص حروف و اعداد فارسی")
    print("="*60)
    
    # ایجاد دیتاست
    train_dataset = PersianPlateDataset(args.data, is_train=True, augment=True)
    val_dataset = PersianPlateDataset(args.data, is_train=False, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PersianPlateDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PersianPlateDataset.collate_fn)
    
    # ایجاد مدل
    model = CRNN(NUM_CLASSES, hidden_size=args.hidden_size).to(DEVICE)
    print_status(f"مدل ایجاد شد - پارامترهای قابل آموزش: {sum(p.numel() for p in model.parameters()):,}", "info")
    
    # آموزش
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, args.epochs, args.lr
    )
    
    # ذخیره مدل نهایی
    torch.save(model.state_dict(), args.save)
    print_status(f"مدل نهایی در {args.save} ذخیره شد", "success")
    
    print("\n" + "="*60)
    print_status("آموزش مدل با موفقیت به پایان رسید!", "success")
    print("="*60)


if __name__ == "__main__":
    main()