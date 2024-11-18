# core/tracker.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import cv2
import numpy as np
from config.settings import Config

# core/tracker.py

@dataclass
class VehicleTrack:
    track_id: int
    license_plate: str = "SCANNING..."
    best_ocr_confidence: float = 0
    processing_complete: bool = False
    state: str = "TRACKING"
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    first_seen_bbox: Optional[Tuple[int, int, int, int]] = None
    ocr_attempts: int = 0
    max_ocr_attempts: int = 2  # Only try OCR twice per vehicle
    best_plate_image: Optional[np.ndarray] = None
    detection_confidence: float = 0
    max_axle_count: int = 2
    frame_path: Optional[str] = None

    def __post_init__(self):
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        
    def update_position(self, bbox: Tuple[int, int, int, int],
                       detection_confidence: float,
                       axle_count: int = 2) -> None:
        """Simple position update"""
        if not self.first_seen_bbox:
            self.first_seen_bbox = bbox
        self.last_bbox = bbox
        self.last_seen = time.time()
        self.detection_confidence = max(self.detection_confidence, detection_confidence)
        self.max_axle_count = max(self.max_axle_count, axle_count)

    def should_process(self, in_roi: bool) -> bool:
        """Only process if we haven't got a good read yet"""
        if self.processing_complete or self.ocr_attempts >= self.max_ocr_attempts:
            return False
        
        return in_roi and self.detection_confidence > 0.5

    def update_plate(self, text: str, confidence: float, plate_image: Optional[np.ndarray] = None) -> bool:
        """Update plate information if better confidence"""
        self.ocr_attempts += 1
        
        if confidence > self.best_ocr_confidence:
            self.best_ocr_confidence = confidence
            self.license_plate = text
            if plate_image is not None:
                self.best_plate_image = plate_image
            
            if confidence > Config.OCR_CONFIDENCE_THRESHOLD:
                self.lock_plate()
                return True
        
        if self.ocr_attempts >= self.max_ocr_attempts:
            self.lock_plate()
        
        return False

    def lock_plate(self) -> None:
        """Lock in the best result we've got"""
        self.processing_complete = True
        self.state = 'LOCKED'

class VehicleTracker:
    def __init__(self):
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_id = 0

    def get_track(self, detection_id: int, bbox: Tuple[int, int, int, int],
                 detection_confidence: float, axle_count: int = 2) -> VehicleTrack:
        """Match detection to existing track or create new one"""
        # Calculate center of new detection
        new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Look for existing tracks
        for track in self.tracks.values():
            if track.last_bbox:
                # Calculate center of existing track
                old_center = ((track.last_bbox[0] + track.last_bbox[2]) / 2,
                            (track.last_bbox[1] + track.last_bbox[3]) / 2)
                
                # Simple distance check
                distance = ((new_center[0] - old_center[0]) ** 2 +
                          (new_center[1] - old_center[1]) ** 2) ** 0.5
                
                if distance < Config.POSITION_THRESHOLD:
                    track.update_position(bbox, detection_confidence, axle_count)
                    return track
        
        # Create new track if no match found
        track = VehicleTrack(track_id=self.next_id)
        track.update_position(bbox, detection_confidence, axle_count)
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def cleanup_old_tracks(self, max_age: float = 3.0):
        """Remove old tracks"""
        current_time = time.time()
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if (current_time - track.last_seen > max_age or
                (track.processing_complete and current_time - track.last_seen > 1.0)):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]

    def draw_tracks(self, frame) -> np.ndarray:
        """Simple visualization of tracks"""
        for track in self.tracks.values():
            if track.last_bbox:
                color = (0, 255, 0) if track.processing_complete else (0, 255, 255)
                
                # Draw bounding box
                cv2.rectangle(frame,
                            (int(track.last_bbox[0]), int(track.last_bbox[1])),
                            (int(track.last_bbox[2]), int(track.last_bbox[3])),
                            color, 2)
                
                # Add text
                text = f"{track.license_plate} ({track.best_ocr_confidence:.1f}%)"
                cv2.putText(frame, text,
                          (int(track.last_bbox[0]), int(track.last_bbox[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add axle count
                cv2.putText(frame, f"Axles: {track.max_axle_count}",
                          (int(track.last_bbox[0]), int(track.last_bbox[1] - 30)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame