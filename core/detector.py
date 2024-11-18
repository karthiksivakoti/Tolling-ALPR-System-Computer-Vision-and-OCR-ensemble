from ultralytics import YOLO
import cv2
import numpy as np
from config.settings import Config
import time
from typing import Dict, List, Tuple, Optional

class VehicleDetector:
    def __init__(self):
        try:
            print("Initializing VehicleDetector...")
            self.model = YOLO(Config.MODEL_PATH)
            print(f"YOLO model loaded from: {Config.MODEL_PATH}")
            self.conf_threshold = 0.5
            
            self.classes = {
                0: 'license_plate',
                1: 'wheel'
            }
            
            # Prevent OpenCV windows
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
            
            print("Detector initialization complete")
        except Exception as e:
            print(f"Error in detector initialization: {str(e)}")
            raise e

    def assign_wheels_to_vehicle(self, plate_bbox: Tuple[int, int, int, int],
                               wheels: List[Dict]) -> int:
        """
        Enhanced wheel/axle counting logic
        Returns total number of axles
        """
        try:
            plate_center_x = (plate_bbox[0] + plate_bbox[2]) / 2
            plate_y = plate_bbox[1]  # Top of license plate
            plate_height = plate_bbox[3] - plate_bbox[1]
            
            # Vehicle bounds estimation (more accurate)
            vehicle_width = (plate_bbox[2] - plate_bbox[0]) * 6  # Estimated vehicle width
            vehicle_left = max(0, plate_center_x - vehicle_width/2)
            vehicle_right = plate_center_x + vehicle_width/2
            
            # Group wheels by vertical position (axles)
            axle_groups = []
            MAX_VERTICAL_DIST = plate_height * 0.5  # Max vertical distance for same axle
            
            for wheel in wheels:
                wheel_bbox = wheel['bbox']
                wheel_center = (
                    (wheel_bbox[0] + wheel_bbox[2]) / 2,
                    (wheel_bbox[1] + wheel_bbox[3]) / 2
                )
                
                # Only consider wheels below plate and within vehicle bounds
                if (wheel_center[1] > plate_y and 
                    vehicle_left <= wheel_center[0] <= vehicle_right):
                    
                    # Try to add to existing axle group
                    added_to_group = False
                    for group in axle_groups:
                        group_y = sum(w[1] for w in group) / len(group)
                        if abs(wheel_center[1] - group_y) < MAX_VERTICAL_DIST:
                            group.append(wheel_center)
                            added_to_group = True
                            break
                    
                    # Create new axle group if needed
                    if not added_to_group:
                        axle_groups.append([wheel_center])
            
            # Count axles (groups with at least one wheel)
            num_axles = len([g for g in axle_groups if len(g) >= 1])
            
            # Ensure minimum of 2 axles
            num_axles = max(2, num_axles)
            
            # Cap at reasonable maximum
            num_axles = min(num_axles, 8)
            
            print(f"Detected {num_axles} axles for vehicle at {plate_bbox}")
            return num_axles
            
        except Exception as e:
            print(f"Error counting axles: {str(e)}")
            return 2  # Default to 2 axles on error

    def detect_and_track(self, frame):
        try:
            print("\nProcessing new frame...")
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,
                show=False  # Ensure no display window
            )
            print("YOLO detection completed")
            
            plates = []
            wheels = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    # Separate plates and wheels
                    for i, box in enumerate(boxes):
                        try:
                            class_id = int(box.cls[0])
                            xyxy = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, xyxy)
                            confidence = float(box.conf[0])
                            
                            detection = {
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class': self.classes[class_id]
                            }
                            
                            if class_id == 0:  # License plate
                                detection['track_id'] = len(plates)
                                plates.append(detection)
                            elif class_id == 1:  # Wheel
                                wheels.append(detection)
                                
                        except Exception as e:
                            print(f"Error processing box: {str(e)}")
                            continue
                    
                    # Apply NMS to plates
                    filtered_plates = self.apply_nms(plates)
                    
                    # Process each plate
                    for plate in filtered_plates:
                        # Count wheels for this plate
                        axle_count = self.assign_wheels_to_vehicle(plate['bbox'], wheels)
                        plate['axle_count'] = axle_count
                        print(f"Vehicle with plate at {plate['bbox']} has {axle_count} axles")
                    
                    return filtered_plates
            
            print("No detections found")
            return []
        
        except Exception as e:
            print(f"Detection Error: {str(e)}")
            return []

    def extract_plate(self, frame, bbox):
        """Extract license plate region from frame"""
        try:
            x1, y1, x2, y2 = bbox
            # Add padding around the plate
            padding = 5
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            plate_img = frame[y1:y2, x1:x2].copy()
            
            # Save debug image
            debug_path = f"debug_plates/raw_plate_{int(time.time()*1000)}.jpg"
            cv2.imwrite(debug_path, plate_img)
            print(f"Saved raw plate to {debug_path}")
            
            return plate_img
        except Exception as e:
            print(f"Plate extraction error: {str(e)}")
            return None

    def apply_nms(self, detections, iou_threshold=0.2):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        try:
            if not detections:
                return []
            
            boxes = []
            scores = []
            for det in detections:
                boxes.append([
                    det['bbox'][0], det['bbox'][1],
                    det['bbox'][2], det['bbox'][3]
                ])
                scores.append(det['confidence'])
            
            boxes = np.array(boxes)
            scores = np.array(scores)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.conf_threshold,
                iou_threshold
            ).flatten()
            
            # Keep only non-overlapping detections
            filtered_detections = [detections[i] for i in indices]
            print(f"NMS: Reduced from {len(detections)} to {len(filtered_detections)} detections")
            
            # Ensure unique track IDs
            for i, det in enumerate(filtered_detections):
                det['track_id'] = i
            
            return filtered_detections
        
        except Exception as e:
            print(f"Error in NMS: {str(e)}")
            return detections