# utils/roi_utils.py
import cv2
import numpy as np
import json
import os
import time
import traceback
from typing import Optional, Tuple, List

class ROIManager:
    def __init__(self, config_path="config/roi_config.json"):
        self.config_path = config_path
        self.roi = self.load_roi()
        self.drawing = False
        self.roi_points = []
        
        # Create debug directory
        os.makedirs("debug_plates", exist_ok=True)
        
        # Prevent OpenCV windows
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    def load_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Load ROI from config file if exists"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                return tuple(data['roi'])
        return None

    def save_roi(self) -> None:
        """Save ROI coordinates to config file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({'roi': list(self.roi)}, f)

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI on frame"""
        if self.roi:
            # Create semi-transparent overlay
            overlay = frame.copy()
            # Darken outside ROI
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                        (0, 0, 0), -1)
            # Clear ROI area
            cv2.rectangle(overlay,
                        (self.roi[0], self.roi[1]),
                        (self.roi[2], self.roi[3]),
                        (0, 0, 0), -1)
            
            # Apply overlay
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw ROI border
            cv2.rectangle(frame,
                        (self.roi[0], self.roi[1]),
                        (self.roi[2], self.roi[3]),
                        (0, 255, 0), 2)
            
            # Draw "SCANNING ZONE" text
            text_size = cv2.getTextSize("SCANNING ZONE",
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, 2)[0]
            
            cv2.putText(frame, "SCANNING ZONE",
                    (self.roi[0], self.roi[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        return frame

    def calculate_intersection(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate intersection percentage of bbox with ROI"""
        try:
            if not self.roi:
                return 0
            
            # Calculate areas
            roi_area = (self.roi[2] - self.roi[0]) * (self.roi[3] - self.roi[1])
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Calculate intersection rectangle
            x1 = max(self.roi[0], bbox[0])
            y1 = max(self.roi[1], bbox[1])
            x2 = min(self.roi[2], bbox[2])
            y2 = min(self.roi[3], bbox[3])
            
            if x1 < x2 and y1 < y2:
                intersection_area = (x2 - x1) * (y2 - y1)
                # Use bbox area as denominator
                intersection_ratio = intersection_area / bbox_area
                
                # Print debug info
                print(f"\nIntersection Debug:")
                print(f"ROI: {self.roi}")
                print(f"BBox: {bbox}")
                print(f"Intersection Area: {intersection_area}")
                print(f"BBox Area: {bbox_area}")
                print(f"Ratio: {intersection_ratio}")
                
                # Save debug visualization without display
                self.save_intersection_debug(bbox, intersection_ratio)
                
                return intersection_ratio
            return 0
                
        except Exception as e:
            print(f"Error calculating intersection: {str(e)}")
            traceback.print_exc()
            return 0

    def save_intersection_debug(self, bbox: Tuple[int, int, int, int], ratio: float) -> None:
        """Save debug visualization of intersection"""
        try:
            # Create a blank image
            debug_img = np.zeros((800, 800, 3), dtype=np.uint8)
            
            # Scale coordinates to fit debug image
            scale = min(700 / max(self.roi[2], bbox[2]), 
                       700 / max(self.roi[3], bbox[3]))
            
            def scale_coords(coords):
                return (
                    int(coords[0] * scale) + 50,
                    int(coords[1] * scale) + 50,
                    int(coords[2] * scale) + 50,
                    int(coords[3] * scale) + 50
                )
            
            roi_scaled = scale_coords(self.roi)
            bbox_scaled = scale_coords(bbox)
            
            # Draw ROI in blue
            cv2.rectangle(debug_img, 
                         (roi_scaled[0], roi_scaled[1]),
                         (roi_scaled[2], roi_scaled[3]),
                         (255, 0, 0), 2)
            
            # Draw bbox in red
            cv2.rectangle(debug_img, 
                         (bbox_scaled[0], bbox_scaled[1]),
                         (bbox_scaled[2], bbox_scaled[3]),
                         (0, 0, 255), 2)
            
            # Draw intersection in green
            x1 = max(roi_scaled[0], bbox_scaled[0])
            y1 = max(roi_scaled[1], bbox_scaled[1])
            x2 = min(roi_scaled[2], bbox_scaled[2])
            y2 = min(roi_scaled[3], bbox_scaled[3])
            
            if x1 < x2 and y1 < y2:
                cv2.rectangle(debug_img,
                            (x1, y1), (x2, y2),
                            (0, 255, 0), -1)
            
            # Add text with intersection ratio
            cv2.putText(debug_img, 
                       f"Intersection Ratio: {ratio:.3f}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (255, 255, 255), 
                       2)
            
            # Save debug image without display
            timestamp = int(time.time()*1000)
            cv2.imwrite(f"debug_plates/roi_intersection_{timestamp}.jpg", debug_img)
            
        except Exception as e:
            print(f"Error saving intersection debug: {str(e)}")

    def get_roi_dimensions(self) -> Tuple[int, int]:
        """Get ROI width and height"""
        if self.roi:
            width = self.roi[2] - self.roi[0]
            height = self.roi[3] - self.roi[1]
            return width, height
        return 0, 0