"""
video_intelligence.py - تحلیل ویدئو، استخراج فریم، ردیابی و تشخیص وقایع
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class VideoIntelligence:
    def init(self, detector, tracker, ocr):
        self.detector = detector
        self.tracker = tracker
        self.ocr = ocr

    def extract_frames(self, video_path: str, interval: int = 30) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    async def process_video_async(self, video_path: str, frame_skip: int = 5) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                plates = self.detector.detect_plates(frame)
                for plate in plates:
                    x1,y1,x2,y2 = map(int, plate['bbox'])
                    crop = frame[y1:y2, x1:x2]
                    text, conf = self.ocr.read_plate(crop)
                    detections.append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'plate_text': text,
                        'confidence': plate['confidence'],
                        'bbox': plate['bbox']
                    })
            frame_idx += 1
        cap.release()
        return {
            'total_frames': total_frames,
            'fps': fps,
            'detections': detections,
            'num_detections': len(detections)
        }

    async def process_video_with_tracking(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        tracks_data = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vehicles = self.detector.detect_vehicles(frame)
            plates = self.detector.detect_plates(frame)
            all_dets = vehicles + plates
            tracks = self.tracker.update(all_dets, frame)
            for track in tracks:
                tracks_data.append({
                    'frame': frame_idx,
                    'track_id': track.track_id,
                    'bbox': track.bbox
                })
            frame_idx += 1
        cap.release()
        return {'tracks': tracks_data, 'unique_tracks': len(set(t['track_id'] for t in tracks_data))}

    def detect_scene_changes(self, video_path: str, threshold: float = 30.0) -> List[int]:
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        scene_changes = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                if mean_diff > threshold:
                    scene_changes.append(frame_idx)
            prev_frame = gray
            frame_idx += 1
        cap.release()
        return scene_changes