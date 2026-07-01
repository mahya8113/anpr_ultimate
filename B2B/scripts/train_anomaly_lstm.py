#!/usr/bin/env python3
"""
train_anomaly_lstm.py
آموزش مدل LSTM برای تشخیص ناهنجاری در مسیر حرکت خودروها
تشخیص: حرکت خلاف جهت، سرعت غیرمجاز، توقف طولانی، مسیر غیرعادی
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import random

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


# ==================== مدل LSTM برای تشخیص ناهنجاری ====================
class LSTMAnomalyDetector(nn.Module):
    """
    مدل LSTM برای تشخیص ناهنجاری در مسیر حرکت خودروها
    """
    def init(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMAnomalyDetector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # لایه LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # لایه‌های Fully Connected
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # خروجی: 0 = نرمال, 1 = ناهنجاری
        
        # Dropout و Activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # استفاده از آخرین خروجی
        last_out = lstm_out[:, -1, :]
        
        # FC لایه‌ها
        out = self.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc3(out))
        
        return out


# ==================== دیتاست ====================
class TrajectoryDataset(Dataset):
    """
    دیتاست مسیر حرکت خودروها برای تشخیص ناهنجاری
    ساختار داده: (seq_len, features) که features شامل [x, y, speed, direction]
    """
    def init(self, trajectories, labels, seq_len=10):
        self.seq_len = seq_len
        self.trajectories = []
        self.labels = []
        
        for traj, label in zip(trajectories, labels):
            if len(traj) >= seq_len:
                # برش به پنجره‌های متوالی
                for i in range(len(traj) - seq_len + 1):
                    self.trajectories.append(traj[i:i+seq_len])
                    self.labels.append(label)
    
    def len(self):
        return len(self.trajectories)
    
    def getitem(self, idx):
        traj = torch.FloatTensor(self.trajectories[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return traj, label


# ==================== تولید داده مصنوعی ====================
def generate_synthetic_data(num_samples=10000, seq_len=10):
    """
    تولید داده مصنوعی برای آموزش مدل تشخیص ناهنجاری
    شامل: مسیرهای نرمال و ناهنجار (حرکت خلاف جهت، سرعت غیرمجاز، توقف طولانی)
    """
    print_status("در حال تولید داده مصنوعی...", "info")
    
    trajectories = []
    labels = []
    
    for _ in range(num_samples):
        # نوع مسیر: 0 = نرمال, 1 = ناهنجار
        is_anomaly = random.random() < 0.3  # 30% ناهنجاری
        
        traj = []
        
        # نقطه شروع تصادفی
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        speed = random.uniform(20, 60)
        direction = random.uniform(0, 360)  # زاویه بر حسب درجه
        
        for t in range(seq_len):
            if is_anomaly:
                # مسیر ناهنجار
                anomaly_type = random.choice(['wrong_direction', 'high_speed', 'sudden_stop', 'zigzag'])
                
                if anomaly_type == 'wrong_direction':
                    # حرکت خلاف جهت (تغییر 180 درجه)
                    if t > seq_len // 2:
                        direction = (direction + 180) % 360
                        speed = random.uniform(30, 60)
                
                elif anomaly_type == 'high_speed':
                    # سرعت غیرمجاز (بیش از 80)
                    if t > seq_len // 3:
                        speed = random.uniform(80, 120)
                
                elif anomaly_type == 'sudden_stop':
                    # توقف ناگهانی
                    if t > seq_len // 2:
                        speed = random.uniform(0, 5)
                
                elif anomaly_type == 'zigzag':
                    # حرکت زیگزاگ
                    direction = direction + random.uniform(-90, 90)
                    speed = random.uniform(40, 70)
            else:
                # مسیر نرمال (حرکت یکنواخت)
                speed = max(20, min(60, speed + random.uniform(-5, 5)))
                direction = direction + random.uniform(-10, 10)
            
            # محاسبه مختصات جدید
            dx = speed * np.cos(np.radians(direction)) * 0.1
            dy = speed * np.sin(np.radians(direction)) * 0.1
            x += dx
            y += dy
            
            # محدود کردن مختصات
            x = max(0, min(100, x))
            y = max(0, min(100, y))
            
            traj.append([x, y, speed, direction])
        
        trajectories.append(np.array(traj))
        labels.append(1 if is_anomaly else 0)
    
    return trajectories, labels


# ==================== بارگذاری داده واقعی ====================
def load_real_data(data_path):
    """
    بارگذاری داده واقعی مسیر خودروها از فایل CSV یا JSON
    فرمت CSV: timestamp, track_id, x, y, speed, direction, label
    """
    if not os.path.exists(data_path):
        print_status(f"فایل {data_path} یافت نشد، از داده مصنوعی استفاده می‌شود", "warning")
        return None, None
    
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            print_status("فرمت فایل پشتیبانی نمی‌شود", "error")
            return None, None
        
        # گروه‌بندی بر اساس track_id
        trajectories = []
        labels = []
        
        for track_id, group in df.groupby('track_id'):
            traj = group[['x', 'y', 'speed', 'direction']].values
            label = group['label'].iloc[0] if 'label' in group.columns else 0
            trajectories.append(traj)
            labels.append(label)
        
        print_status(f"داده واقعی بارگذاری شد: {len(trajectories)} مسیر", "success")
        return trajectories, labels
        
    except Exception as e:
        print_status(f"خطا در بارگذاری داده: {e}", "error")
        return None, None


# ==================== نرمال‌سازی داده ====================
def normalize_data(trajectories, scaler=None):
    """نرمال‌سازی ویژگی‌ها"""
    all_features = np.concatenate([traj for traj in trajectories], axis=0)
    
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(all_features)
    
    normalized_trajs = []
    for traj in trajectories:
        traj_flat = traj.reshape(-1, traj.shape[-1])
        normalized = scaler.transform(traj_flat)
        normalized_trajs.append(normalized.reshape(traj.shape))
    
    return normalized_trajs, scaler


# ==================== آموزش مدل ====================
def train_model(model, train_loader, val_loader, epochs, lr=0.001):
    """آموزش مدل LSTM"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ========== آموزش ==========
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # ========== اعتبارسنجی ==========
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        # ذخیره بهترین مدل
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/lstm_anomaly_best.pt')
        
        if (epoch + 1) % 10 == 0:
            print_status(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", "info")
    
    return train_losses, val_losses, train_accs, val_accs


# ==================== ارزیابی مدل ====================
def evaluate_model(model, test_loader):
    """ارزیابی مدل روی داده تست"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(DEVICE)
            outputs = model(data)
            predicted = (outputs > 0.5).cpu().numpy()
            all_preds.extend(predicted.flatten())
            all_labels.extend(labels.numpy().flatten())
    
    print("\n" + "="*50)
    print("📊 گزارش ارزیابی مدل")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Anomaly']))
    # ماتریس درهم‌ریختگی
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix - تشخیص ناهنجاری')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/lstm_anomaly_confusion_matrix.png')
    plt.close()
    
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nدقت نهایی مدل:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return accuracy


# ==================== رسم نمودارهای آموزش ====================
def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    """رسم نمودارهای Loss و Accuracy"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # نمودار Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # نمودار Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/lstm_anomaly_training_history.png')
    plt.close()
    print_status("نمودارهای آموزش در 'models/lstm_anomaly_training_history.png' ذخیره شد", "success")


# ==================== تابع پیش‌بینی ====================
def predict_anomaly(model, trajectory, scaler, seq_len=10):
    """
    پیش‌بینی ناهنجاری برای یک مسیر جدید
    """
    model.eval()
    
    # نرمال‌سازی
    traj_normalized = scaler.transform(trajectory)
    
    # ایجاد پنجره‌های متوالی
    predictions = []
    for i in range(len(traj_normalized) - seq_len + 1):
        window = torch.FloatTensor(traj_normalized[i:i+seq_len]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(window).cpu().numpy()[0][0]
            predictions.append(pred)
    
    # میانگین پیش‌بینی‌ها
    avg_pred = np.mean(predictions)
    return {
        "is_anomaly": avg_pred > 0.5,
        "anomaly_score": float(avg_pred),
        "frame_predictions": predictions
    }


# ==================== تابع اصلی ====================
def main():
    parser = argparse.ArgumentParser(description='آموزش مدل LSTM برای تشخیص ناهنجاری در مسیر حرکت خودروها')
    parser.add_argument('--data', type=str, help='مسیر فایل داده واقعی (CSV/JSON)')
    parser.add_argument('--epochs', type=int, default=50, help='تعداد دوره‌های آموزش')
    parser.add_argument('--batch-size', type=int, default=32, help='سایز دسته')
    parser.add_argument('--seq-len', type=int, default=10, help='طول دنباله زمانی')
    parser.add_argument('--hidden-size', type=int, default=128, help='سایز لایه پنهان LSTM')
    parser.add_argument('--lr', type=float, default=0.001, help='نرخ یادگیری')
    parser.add_argument('--save', type=str, default='models/lstm_anomaly.pt', help='مسیر ذخیره مدل')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚗 آموزش مدل LSTM برای تشخیص ناهنجاری")
    print("="*60)
    
    # بارگذاری داده
    if args.data:
        trajectories, labels = load_real_data(args.data)
        if not args.data or trajectories is None:
         print_status("تولید داده مصنوعی...", "info")
        trajectories, labels = generate_synthetic_data(num_samples=5000, seq_len=args.seq_len)
    
    print_status(f"تعداد مسیرها: {len(trajectories)}", "info")
    print_status(f"تعداد ناهنجاری‌ها: {sum(labels)}", "info")
    
    # نرمال‌سازی داده
    trajectories, scaler = normalize_data(trajectories)
    
    # تقسیم داده به train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(trajectories, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print_status(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}", "info")
    
    # ایجاد دیتاست
    train_dataset = TrajectoryDataset(X_train, y_train, args.seq_len)
    val_dataset = TrajectoryDataset(X_val, y_val, args.seq_len)
    test_dataset = TrajectoryDataset(X_test, y_test, args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print_status(f"تعداد نمونه‌های آموزش: {len(train_dataset)}", "info")
    
    # ایجاد مدل
    model = LSTMAnomalyDetector(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=0.3
    ).to(DEVICE)
    
    print_status(f"مدل ایجاد شد - پارامترهای قابل آموزش: {sum(p.numel() for p in model.parameters()):,}", "info")
    
    # آموزش مدل
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, args.epochs, args.lr
    )
    
    # ارزیابی مدل
    accuracy = evaluate_model(model, test_loader)
    
    # رسم نمودارها
    plot_training_results(train_losses, val_losses, train_accs, val_accs)
    
    # ذخیره مدل نهایی
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': {
            'input_size': 4,
            'hidden_size': args.hidden_size,
            'num_layers': 2,
            'seq_len': args.seq_len,
            'accuracy': accuracy
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    }, args.save)
    
    print_status(f"مدل نهایی در {args.save} ذخیره شد", "success")
    
    # ذخیره scaler
    import joblib
    joblib.dump(scaler, 'models/lstm_anomaly_scaler.pkl')
    print_status("StandardScaler در 'models/lstm_anomaly_scaler.pkl' ذخیره شد", "success")
    
    print("\n" + "="*60)
    print_status("آموزش مدل با موفقیت به پایان رسید!", "success")
    print("="*60)


if __name__ == "__main__":
    main()