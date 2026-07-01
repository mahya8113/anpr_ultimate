"""
anomaly.py - تشخیص ناهنجاری در رفتار خودروها
شامل: تشخیص حرکت خلاف جهت، سرعت غیرمجاز، توقف طولانی، مسیر غیرعادی، و ناهنجاری‌های زمانی
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ==================== مدل LSTM برای تشخیص ناهنجاری ====================
class LSTMAnomalyDetector(nn.Module):
    """
    مدل LSTM برای تشخیص ناهنجاری در دنباله‌های حرکتی
    ورودی: دنباله‌ای از موقعیت‌ها (x, y, سرعت, زاویه)
    خروجی: نمره ناهنجاری (0 تا 1)
    """
    def init(self, input_size: int = 4, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().init()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        # استفاده از آخرین خروجی
        last_out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out


# ==================== کلاس‌های داده ====================
@dataclass
class TrajectoryPoint:
    """نقطه مسیر خودرو"""
    x: float
    y: float
    speed: float
    direction: float  # زاویه بر حسب درجه
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Anomaly:
    """نتیجه تشخیص ناهنجاری"""
    type: str
    severity: str  # 'low', 'medium', 'high'
    track_id: int
    plate_text: Optional[str]
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


# ==================== تشخیص ناهنجاری بر اساس قوانین ====================
class RuleBasedAnomalyDetector:
    """
    تشخیص ناهنجاری بر اساس قوانین از پیش تعریف شده
    شامل: سرعت غیرمجاز، حرکت خلاف جهت، توقف طولانی، خروج از منطقه
    """
    
    def init(self, config: Optional[Dict] = None):
        """
        Args:
            config: تنظیمات مانند speed_limit، wrong_direction_threshold و ...
        """
        self.config = config or {
            "speed_limit_ms": 16.7,  # 60 km/h
            "wrong_direction_threshold": 90,  # درجه اختلاف جهت مجاز
            "stop_duration_seconds": 30,  # توقف طولانی بعد از این مدت
            "stop_speed_threshold": 0.5,  # سرعت کمتر از این به عنوان توقف در نظر گرفته شود
            "max_acceleration": 5.0,  # شتاب غیرعادی (m/s²)
            "zone_limits": None  # [(x1,y1,x2,y2), ...] مناطق مجاز
        }
    
    def check_speed(self, speed: float) -> Tuple[bool, str, float]:
        """بررسی سرعت غیرمجاز"""
        if speed > self.config["speed_limit_ms"]:
            severity = "high" if speed > self.config["speed_limit_ms"] * 1.5 else "medium"
            return True, f"سرعت غیرمجاز: {speed * 3.6:.1f} km/h", severity
        return False, "", "low"
    
    def check_direction_change(self, prev_dir: float, curr_dir: float) -> Tuple[bool, str, float]:
        """بررسی تغییر جهت ناگهانی (حرکت خلاف جهت)"""
        diff = abs(curr_dir - prev_dir)
        diff = min(diff, 360 - diff)
        if diff > self.config["wrong_direction_threshold"]:
            return True, f"تغییر جهت ناگهانی: {diff:.0f} درجه", "medium"
        return False, "", "low"
    
    def check_sudden_stop(self, prev_speed: float, curr_speed: float, dt: float) -> Tuple[bool, str, float]:
        """بررسی توقف ناگهانی"""
        if dt > 0:
            deceleration = (prev_speed - curr_speed) / dt
            if deceleration > self.config["max_acceleration"] and curr_speed < 1.0:
                return True, f"توقف ناگهانی با شتاب {-deceleration:.1f} m/s²", "medium"
        return False, "", "low"
    
    def check_prolonged_stop(self, track_history: List[TrajectoryPoint]) -> Tuple[bool, str, float]:
        """بررسی توقف طولانی مدت"""
        if len(track_history) < 2:
            return False, "", "low"
        
        # یافتن آخرین نقطه با سرعت بالا
        last_moving_idx = len(track_history) - 1
        for i in range(len(track_history) - 1, -1, -1):
            if track_history[i].speed > self.config["stop_speed_threshold"]:
                last_moving_idx = i
                break
        
        if last_moving_idx < len(track_history) - 1:
            stop_duration = (track_history[-1].timestamp - track_history[last_moving_idx].timestamp).total_seconds()
            if stop_duration > self.config["stop_duration_seconds"]:
                return True, f"توقف طولانی: {stop_duration:.0f} ثانیه", "low"
        
        return False, "", "low"
    
    def check_zone_violation(self, x: float, y: float) -> Tuple[bool, str, float]:
        """بررسی خروج از منطقه مجاز"""
        if self.config["zone_limits"]:
            in_zone = any(x1 <= x <= x2 and y1 <= y <= y2 for (x1, y1, x2, y2) in self.config["zone_limits"])
            if not in_zone:
                return True, f"خروج از محدوده مجاز: ({x:.1f}, {y:.1f})", "high"
        return False, "", "low"
    
    def detect(self, track_id: int, plate_text: Optional[str], history: List[TrajectoryPoint]) -> List[Anomaly]:
        """تشخیص ناهنجاری‌ها بر اساس قوانین"""
        anomalies = []
        
        if len(history) < 2:
            return anomalies
        
        curr = history[-1]
        prev = history[-2]
        dt = (curr.timestamp - prev.timestamp).total_seconds()
        if dt <= 0:
            dt = 0.1
        
        # بررسی سرعت
        is_anomaly, msg, sev = self.check_speed(curr.speed)
        if is_anomaly:
            anomalies.append(Anomaly(
                type="speeding",
                severity=sev,
                track_id=track_id,
                plate_text=plate_text,
                message=msg,
                timestamp=curr.timestamp,
                details={"speed_kph": curr.speed * 3.6, "limit_kph": self.config["speed_limit_ms"] * 3.6}
            ))
        
        # بررسی تغییر جهت ناگهانی
        is_anomaly, msg, sev = self.check_direction_change(prev.direction, curr.direction)
        if is_anomaly:
            anomalies.append(Anomaly(
                type="wrong_direction",
                severity=sev,
                track_id=track_id,
                plate_text=plate_text,
                message=msg,
                timestamp=curr.timestamp,
                details={"direction_change": abs(curr.direction - prev.direction)}
            ))
        
        # بررسی توقف ناگهانی
        is_anomaly, msg, sev = self.check_sudden_stop(prev.speed, curr.speed, dt)
        if is_anomaly:
            anomalies.append(Anomaly(
                type="sudden_stop",
                severity=sev,
                track_id=track_id,
                plate_text=plate_text,
                message=msg,
                timestamp=curr.timestamp,
                details={"deceleration": (prev.speed - curr.speed) / dt}
            ))
        
        # بررسی توقف طولانی
        is_anomaly, msg, sev = self.check_prolonged_stop(history)
        if is_anomaly:
            anomalies.append(Anomaly(
                type="prolonged_stop",
                severity=sev,
                track_id=track_id,
                plate_text=plate_text,
                message=msg,
                timestamp=curr.timestamp,
                details={}
            ))
        
        # بررسی خروج از منطقه
        is_anomaly, msg, sev = self.check_zone_violation(curr.x, curr.y)
        if is_anomaly:
            anomalies.append(Anomaly(
                type="zone_violation",
                severity=sev,
                track_id=track_id,
                plate_text=plate_text,
                message=msg,
                timestamp=curr.timestamp,
                details={"position": (curr.x, curr.y)}
            ))
        
        return anomalies


# ==================== تشخیص ناهنجاری با LSTM ====================
class LSTMAnomalyDetectorWrapper:
    """
    wrapper برای استفاده از مدل LSTM در تشخیص ناهنجاری
    نیاز به آموزش قبلی مدل دارد
    """
    
    def init(self, model_path: Optional[str] = None, device: str = 'cuda', seq_len: int = 10):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.seq_len = seq_len
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """بارگذاری مدل آموزش دیده"""
        self.model = LSTMAnomalyDetector().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"مدل LSTM از {model_path} بارگذاری شد")
    
    def prepare_sequence(self, history: List[TrajectoryPoint]) -> Optional[torch.Tensor]:
        """آماده‌سازی دنباله برای ورودی به LSTM"""
        if len(history) < self.seq_len:
            return None
        
        # استخراج ویژگی‌ها: [x, y, speed, direction]
        features = []
        for p in history[-self.seq_len:]:
            features.append([p.x, p.y, p.speed, p.direction])
        
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, history: List[TrajectoryPoint]) -> Tuple[float, bool]:
        """
        پیش‌بینی نمره ناهنجاری
        
        Returns:
            (anomaly_score, is_anomaly) — is_anomaly اگر score > 0.5
        """
        if self.model is None:
            return 0.0, False
        
        seq = self.prepare_sequence(history)
        if seq is None:
            return 0.0, False
        
        with torch.no_grad():
            score = self.model(seq).item()
        
        return score, score > 0.5


# ==================== تشخیص ناهنجاری ترکیبی ====================
class HybridAnomalyDetector:
    """
    تشخیص ناهنجاری با ترکیب روش‌های قانونی و یادگیری عمیق
    """
    
    def init(self, config: Optional[Dict] = None, lstm_model_path: Optional[str] = None):
        self.rule_detector = RuleBasedAnomalyDetector(config)
        self.lstm_detector = None
        if lstm_model_path:
            self.lstm_detector = LSTMAnomalyDetectorWrapper(lstm_model_path)
        
        self.track_histories: Dict[int, List[TrajectoryPoint]] = {}
        self.track_plate_map: Dict[int, str] = {}
    
    def update_track(self, track_id: int, x: float, y: float, speed: float, direction: float, plate_text: Optional[str] = None):
        """به‌روزرسانی تاریخچه مسیر یک خودرو"""
        point = TrajectoryPoint(
            x=x, y=y, speed=speed, direction=direction,
            timestamp=datetime.now()
        )
        
        if track_id not in self.track_histories:
            self.track_histories[track_id] = []
        
        self.track_histories[track_id].append(point)
        
        # محدود کردن طول تاریخچه
        max_history = 300
        if len(self.track_histories[track_id]) > max_history:
            self.track_histories[track_id] = self.track_histories[track_id][-max_history:]
        
        if plate_text:
            self.track_plate_map[track_id] = plate_text
    
    def detect(self, track_id: int) -> List[Anomaly]:
        """تشخیص ناهنجاری برای یک خودرو"""
        if track_id not in self.track_histories:
            return []
        
        history = self.track_histories[track_id]
        plate = self.track_plate_map.get(track_id)
        
        # تشخیص با قوانین
        rule_anomalies = self.rule_detector.detect(track_id, plate, history)
        
        # تشخیص با LSTM (در صورت وجود)
        lstm_anomalies = []
        if self.lstm_detector and len(history) >= self.lstm_detector.seq_len:
            score, is_anomaly = self.lstm_detector.predict(history)
            if is_anomaly:
                lstm_anomalies.append(Anomaly(
                    type="lstm_anomaly",
                    severity="medium",
                    track_id=track_id,
                    plate_text=plate,
                    message=f"ناهنجاری تشخیص داده شده توسط LSTM (نمره: {score:.2f})",
                    timestamp=datetime.now(),
                    details={"lstm_score": score}
                ))
        
        return rule_anomalies + lstm_anomalies
    
    def detect_all_active(self) -> Dict[int, List[Anomaly]]:
        """تشخیص ناهنجاری برای همه خودروهای فعال"""
        results = {}
        for track_id in self.track_histories.keys():
            anomalies = self.detect(track_id)
            if anomalies:
                results[track_id] = anomalies
        return results
    
    def remove_old_tracks(self, max_idle_seconds: int = 60):
        """حذف خودروهایی که مدت زیادی دیده نشده‌اند"""
        now = datetime.now()
        to_remove = []
        for track_id, history in self.track_histories.items():
            if history and (now - history[-1].timestamp).total_seconds() > max_idle_seconds:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_histories[track_id]
            if track_id in self.track_plate_map:
                del self.track_plate_map[track_id]
        
        if to_remove:
            logger.info(f"حذف {len(to_remove)} track قدیمی")
    
    def get_statistics(self) -> Dict:
        """دریافت آمار"""
        return {
            "active_tracks": len(self.track_histories),
            "tracks_with_plate": len(self.track_plate_map),
            "rule_based_active": True,
            "lstm_available": self.lstm_detector is not None
        }


# ==================== توابع کمکی ====================

def calculate_speed_and_direction(x1, y1, x2, y2, dt: float) -> Tuple[float, float]:
    """محاسبه سرعت (m/s) و زاویه (درجه) بین دو نقطه"""
    dx = x2 - x1
    dy = y2 - y1
    distance = np.hypot(dx, dy)
    speed = distance / dt if dt > 0 else 0
    direction = np.degrees(np.arctan2(dy, dx))
    return speed, direction


def create_anomaly_report(anomalies: List[Anomaly]) -> Dict:
    """ایجاد گزارش از ناهنجاری‌ها"""
    return {
        "total_anomalies": len(anomalies),
        "by_type": {t: len([a for a in anomalies if a.type == t]) for t in set(a.type for a in anomalies)},
        "by_severity": {s: len([a for a in anomalies if a.severity == s]) for s in ['low', 'medium', 'high']},
        "anomalies": [
            {
                "type": a.type,
                "severity": a.severity,
                "track_id": a.track_id,
                "plate": a.plate_text,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
                "details": a.details
            }
            for a in anomalies
        ]
    }


# ==================== مثال استفاده ====================
if __name__ == "main":
    # تست سریع
    detector = HybridAnomalyDetector()
    
    # شبیه‌سازی چند نقطه
    detector.update_track(1, 0, 0, 10, 90)
    detector.update_track(1, 10, 0, 20, 90)
    detector.update_track(1, 20, 0, 25, 90)
    detector.update_track(1, 30, 0, 30, 90)  # سرعت بالا
    
    anomalies = detector.detect(1)
    for a in anomalies:
        print(f"ناهنجاری: {a.type} - {a.message} (شدت: {a.severity})")
        