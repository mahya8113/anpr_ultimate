"""
multi_tracking.py - ردیابی چند شیء با استفاده از DeepSORT
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import cv2
logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("DeepSORT not available. Install: pip install deep-sort-realtime")


@dataclass
class Track:
    track_id: int
    bbox: List[int]
    class_id: int
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    plate_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTracker:
    def init(self, max_age: int = 30, n_init: int = 3, nn_budget: int = 100, use_custom_model: bool = False):
        self.max_age = max_age
        self.n_init = n_init
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget,
                                    embedder="mobilenet" if not use_custom_model else "custom")
        else:
            self.tracker = None
        self.tracks: Dict[int, Track] = {}
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.entry_zones = []
        self.exit_zones = []

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Track]:
        if self.tracker is None:
            return self._manual_update(detections)
        deepsort_dets = []
        for det in detections:
            bbox = det['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            conf = det.get('confidence', 0.5)
            cls = det.get('class_id', 0)
            deepsort_dets.append(([bbox[0], bbox[1], w, h], conf, cls))
        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
            if tid in self.tracks:
                old = self.tracks[tid]
                old.bbox = bbox
                old.age = track.age
                old.hits = track.hits
                old.time_since_update = track.time_since_update
                old.trajectory.append(center)
                old.last_seen = datetime.now()
                results.append(old)
            else:
                new = Track(track_id=tid, bbox=bbox, class_id=0, confidence=track.get_det_conf() or 0.5,
                            age=track.age, hits=track.hits, trajectory=[center])
                self.tracks[tid] = new
                results.append(new)
            self.track_history.setdefault(tid, []).append(center)
        # remove old tracks
        to_remove = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]
        return results

    def _manual_update(self, detections: List[Dict]) -> List[Track]:
        results = []
        for det in detections:
            tid = len(self.tracks) + 1
            bbox = det['bbox']
            center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
            track = Track(track_id=tid, bbox=bbox, class_id=det.get('class_id',0), confidence=det.get('confidence',0.5), trajectory=[center])
            self.tracks[tid] = track
            results.append(track)
        return results

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[Track]:
        return list(self.tracks.values())
    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        return self.track_history.get(track_id, [])

    def set_entry_zones(self, zones: List[List[int]]):
        self.entry_zones = zones

    def set_exit_zones(self, zones: List[List[int]]):
        self.exit_zones = zones

    def check_entry_exit(self, track: Track) -> Tuple[bool, bool]:
        if not track.trajectory:
            return False, False
        center = track.trajectory[-1]
        is_entry = any(self._point_in_zone(center, z) for z in self.entry_zones)
        is_exit = any(self._point_in_zone(center, z) for z in self.exit_zones)
        return is_entry, is_exit

    def _point_in_zone(self, point: Tuple[int, int], zone: List[int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    def assign_plate(self, track_id: int, plate_text: str):
        if track_id in self.tracks:
            self.tracks[track_id].plate_text = plate_text

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        img = frame.copy()
        for track in self.tracks.values():
            if len(track.trajectory) >= 2:
                pts = np.array(track.trajectory[-30:], dtype=np.int32)
                cv2.polylines(img, [pts], False, (0,255,0), 2)
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            label = f"ID:{track.track_id}"
            if track.plate_text: label += f" {track.plate_text}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        for zone in self.entry_zones:
            x1,y1,x2,y2 = zone
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, "ENTRY", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        for zone in self.exit_zones:
            x1,y1,x2,y2 = zone
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, "EXIT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return img