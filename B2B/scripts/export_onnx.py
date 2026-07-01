#!/usr/bin/env python3
"""
export_onnx.py
اسکریپت تبدیل مدل‌های PyTorch (.pt) به فرمت ONNX برای استقرار روی Edge devices و CPU
پشتیبانی از: YOLOv8، CRNN، LSTM و سایر مدل‌ها
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import onnx
import onnxruntime
from typing import Optional, Tuple
import json

# ==================== رنگ‌ها برای خروجی زیبا ====================
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
    elif status == "info":
        print(f"{BLUE}📌 {message}{RESET}")
    else:
        print(f"📌 {message}")


# ==================== تابع اعتبارسنجی ONNX ====================
def validate_onnx_model(onnx_path: str, verbose: bool = True) -> bool:
    """اعتبارسنجی فایل ONNX"""
    try:
        # بارگذاری مدل ONNX
        model = onnx.load(onnx_path)
        
        # اعتبارسنجی
        onnx.checker.check_model(model)
        if verbose:
            print_status(f"مدل {onnx_path} معتبر است", "success")
            
        # نمایش اطلاعات مدل
        if verbose:
            print(f"  نسخه ONNX: {model.opset_import[0].version if model.opset_import else 'نامشخص'}")
            print(f"  ورودی‌ها: {len(model.graph.input)}")
            for inp in model.graph.input:
                print(f"    - {inp.name}")
            print(f"  خروجی‌ها: {len(model.graph.output)}")
            
        return True
    except Exception as e:
        print_status(f"خطا در اعتبارسنجی مدل: {e}", "error")
        return False


# ==================== تست اجرای ONNX ====================
def test_onnx_inference(onnx_path: str, input_shape: Tuple[int, ...]) -> bool:
    """تست اجرای مدل ONNX با یک ورودی نمونه"""
    try:
        # ایجاد session
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # ایجاد ورودی تصادفی
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # اجرا
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: dummy_input})
        
        print_status("تست اجرای ONNX موفق بود", "success")
        print(f"  شکل خروجی: {outputs[0].shape}")
        return True
    except Exception as e:
        print_status(f"خطا در تست اجرا: {e}", "error")
        return False


# ==================== 1. تبدیل مدل YOLOv8 ====================
def export_yolov8_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True
) -> Optional[str]:
    """
    تبدیل مدل YOLOv8 به ONNX
    
    Parameters:
    -----------
    model_path : str
        مسیر فایل .pt مدل YOLO
    output_path : str
        مسیر خروجی (اختیاری)
    imgsz : int
        اندازه تصویر ورودی
    half : bool
        استفاده از FP16
    simplify : bool
        ساده‌سازی مدل ONNX
    """
    print("\n" + "="*50)
    print("🔄 1. تبدیل مدل YOLOv8 به ONNX")
    print("="*50)
    
    try:
        from ultralytics import YOLO
        
        # بارگذاری مدل
        print_status(f"بارگذاری مدل از {model_path}", "info")
        model = YOLO(model_path)
        
        # تبدیل
        if output_path is None:
            output_path = model_path.replace('.pt', '.onnx')
        
        print_status(f"در حال تبدیل به {output_path}...", "info")
        
        # پارامترهای تبدیل
        model.export(
            format='onnx',
            imgsz=imgsz,
            half=half,
            simplify=simplify,
            opset=12,
            dynamic=False  # برای Edge بهتر است false باشد
        )
        
        # YOLO به طور خودکار فایل را در پوشه کناری ذخیره می‌کند
        default_output = str(Path(model_path).parent / f"{Path(model_path).stem}.onnx")
        if os.path.exists(default_output):
            if output_path != default_output:
                import shutil
                shutil.move(default_output, output_path)
        
        if os.path.exists(output_path):
            print_status(f"مدل YOLO با موفقیت به {output_path} تبدیل شد", "success")
            
            # اعتبارسنجی
            validate_onnx_model(output_path)
            test_onnx_inference(output_path, (1, 3, imgsz, imgsz))
            
            return output_path
        else:
            print_status("فایل خروجی پیدا نشد", "error")
            return None
            
    except ImportError:
        print_status("کتابخانه ultralytics نصب نیست. برای نصب: pip install ultralytics", "error")
        return None
    except Exception as e:
        print_status(f"خطا در تبدیل YOLO: {e}", "error")
        return None


# ==================== 2. تبدیل مدل CRNN به ONNX ====================
def export_crnn_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    input_height: int = 48,
    input_width: int = 160,
    num_classes: int = 35,
    hidden_size: int = 256
) -> Optional[str]:
    """
    تبدیل مدل CRNN (تشخیص حروف فارسی) به ONNX
    
    Parameters:
    -----------
    model_path : str
        مسیر فایل .pth مدل CRNN
    output_path : str
        مسیر خروجی
    input_height : int
        ارتفاع ورودی تصویر
    input_width : int
        عرض ورودی تصویر
    num_classes : int
        تعداد کلاس‌ها (حروف + اعداد + blank)
    hidden_size : int
        اندازه لایه hidden در LSTM
    """
    print("\n" + "="*50)
    print("🔄 2. تبدیل مدل CRNN به ONNX")
    print("="*50)
    
    try:
        import torch.nn as nn
        
        # تعریف معماری CRNN (باید با مدل اصلی یکسان باشد)
        class CRNN(nn.Module):
            def init(self, num_classes, hidden_size=256):
                super(CRNN, self).__init__()
                
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1), (2, 1)),
                )
                
                self.rnn = nn.LSTM(
                    input_size=512 * (input_height // 16),
                    hidden_size=hidden_size,
                    num_layers=2,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.5
                )
                
                self.fc = nn.Linear(hidden_size * 2, num_classes)
            
            def forward(self, x):
                batch_size = x.size(0)
                features = self.cnn(x)
                features = features.permute(0, 3, 1, 2)
                features = features.reshape(batch_size, -1, features.size(1) * features.size(2))
                rnn_out, _ = self.rnn(features)
                output = self.fc(rnn_out)
                return output
        
        # بارگذاری مدل
        print_status(f"بارگذاری مدل CRNN از {model_path}", "info")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CRNN(num_classes, hidden_size).to(device)
        
        # بارگذاری وزن‌ها
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print_status("وزن‌های مدل با موفقیت بارگذاری شد", "success")
        else:
            print_status(f"فایل {model_path} یافت نشد، ایجاد مدل با وزن‌های تصادفی", "warning")
        
        model.eval()
        
        # ورودی نمونه برای تبدیل
        dummy_input = torch.randn(1, 1, input_height, input_width).to(device)
        
        # خروجی
        if output_path is None:
            output_path = model_path.replace('.pth', '.onnx') if model_path.endswith('.pth') else 'models/crnn_model.onnx'
        
        # تبدیل
        print_status(f"در حال تبدیل به {output_path}...", "info")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        if os.path.exists(output_path):
            print_status(f"مدل CRNN با موفقیت به {output_path} تبدیل شد", "success")
            validate_onnx_model(output_path)
            test_onnx_inference(output_path, (1, 1, input_height, input_width))
            return output_path
        else:
            return None
            
    except Exception as e:
        print_status(f"خطا در تبدیل CRNN: {e}", "error")
        return None


# ==================== 3. تبدیل مدل LSTM (Anomaly Detection) به ONNX ====================
def export_lstm_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    input_size: int = 4,
    hidden_size: int = 64,
    num_layers: int = 2,
    seq_len: int = 10
) -> Optional[str]:
    """
    تبدیل مدل LSTM برای تشخیص ناهنجاری به ONNX
    """
    print("\n" + "="*50)
    print("🔄 3. تبدیل مدل LSTM به ONNX")
    print("="*50)
    
    try:
        import torch.nn as nn
        
        class LSTMAnomaly(nn.Module):
            def init(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, input_size)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        
        # بارگذاری مدل
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMAnomaly(input_size, hidden_size, num_layers).to(device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print_status("وزن‌های مدل با موفقیت بارگذاری شد", "success")
        
        model.eval()
        
        # ورودی نمونه
        dummy_input = torch.randn(1, seq_len, input_size).to(device)
        
        # خروجی
        if output_path is None:
            output_path = model_path.replace('.pt', '.onnx') if model_path.endswith('.pt') else 'models/lstm_model.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'seq_len'},
                'output': {0: 'batch_size'}
            }
        )
        
        if os.path.exists(output_path):
            print_status(f"مدل LSTM با موفقیت به {output_path} تبدیل شد", "success")
            validate_onnx_model(output_path)
            return output_path
        return None
        
    except Exception as e:
        print_status(f"خطا در تبدیل LSTM: {e}", "error")
        return None


# ==================== 4. تبدیل کل مدل‌ها به ONNX (Batch Export) ====================
def export_all_models(config_file: Optional[str] = None):
    """
    تبدیل تمام مدل‌های پروژه به ONNX
    """
    print("\n" + "="*60)
    print("🚀 تبدیل تمام مدل‌های پروژه به فرمت ONNX")
    print("="*60)
    
    results = {}
    
    # مدل YOLO
    yolo_paths = [
        "models/yolov8n.pt",
        "models/yolov8n_plate_v1.pt",
        "models/yolov8s.pt"
    ]
    
    for path in yolo_paths:
        if os.path.exists(path):
            onnx_path = export_yolov8_to_onnx(path)
            if onnx_path:
                results[path] = onnx_path
    
    # مدل CRNN
    crnn_path = "models/crnn_persian_v1.pth"
    if os.path.exists(crnn_path):
        onnx_path = export_crnn_to_onnx(crnn_path)
        if onnx_path:
            results[crnn_path] = onnx_path
    
    # مدل LSTM
    lstm_path = "models/lstm_anomaly.pt"
    if os.path.exists(lstm_path):
        onnx_path = export_lstm_to_onnx(lstm_path)
        if onnx_path:
            results[lstm_path] = onnx_path
    
    # ذخیره گزارش
    report_path = "models/onnx_export_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print_status("گزارش تبدیل ONNX ذخیره شد", "success")
    print(f"📄 {report_path}")
    
    return results


# ==================== تابع اصلی ====================
def main():
    parser = argparse.ArgumentParser(description='تبدیل مدل‌های PyTorch به ONNX برای استقرار روی Edge')
    parser.add_argument('--model', type=str, help='مسیر فایل مدل (.pt یا .pth)')
    parser.add_argument('--type', type=str, choices=['yolo', 'crnn', 'lstm', 'auto'], 
                        default='auto', help='نوع مدل')
    parser.add_argument('--output', type=str, help='مسیر خروجی (اختیاری)')
    parser.add_argument('--imgsz', type=int, default=640, help='اندازه تصویر برای YOLO')
    parser.add_argument('--half', action='store_true', help='استفاده از FP16')
    parser.add_argument('--all', action='store_true', help='تبدیل تمام مدل‌های موجود')
    
    args = parser.parse_args()
    
    if args.all:
        export_all_models()
        return
    
    if not args.model:
        print_status("لطفاً مسیر فایل مدل را با --model مشخص کنید", "error")
        print("مثال: python export_onnx.py --model models/yolov8n.pt --type yolo")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print_status(f"فایل {args.model} یافت نشد", "error")
        sys.exit(1)
    
    # تشخیص خودکار نوع مدل
    if args.type == 'auto':
        if 'yolo' in args.model.lower():
            args.type = 'yolo'
        elif 'crnn' in args.model.lower() or '.pth' in args.model:
            args.type = 'crnn'
        elif 'lstm' in args.model.lower() or 'anomaly' in args.model.lower():
            args.type = 'lstm'
        else:
            print_status("نوع مدل قابل تشخیص نیست، لطفاً با --type مشخص کنید", "error")
            sys.exit(1)
    
    # تبدیل مدل
    if args.type == 'yolo':
        export_yolov8_to_onnx(args.model, args.output, args.imgsz, args.half)
    elif args.type == 'crnn':
        export_crnn_to_onnx(args.model, args.output)
    elif args.type == 'lstm':
        export_lstm_to_onnx(args.model, args.output)


if __name__ == "__main__":
    main()