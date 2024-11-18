# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    TESSERACT_PATH = os.getenv('TESSERACT_PATH')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')
    DATABASE_URL = "sqlite:///license_plate_system.db"
    
    # OCR Settings
    OCR_CONFIDENCE_THRESHOLD = 40  # Minimum confidence for valid plate
    ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    # Detection Settings
    DETECTION_CONFIDENCE_THRESHOLD = 0.4  # Lowered to catch more plates0.6
    MIN_PLATE_LENGTH = 3  # Minimum number of characters in plate
    MAX_PLATE_LENGTH = 10  # Maximum number of characters in plate
    NMS_THRESHOLD = 0.3  # NMS threshold for removing duplicate detections
    
    
    # Plate Processing Thresholds
    SAVE_THRESHOLD = 75.0  # Save and process plate above this confidence60
    LOCK_THRESHOLD = 80.0  # Lock plate reading above this confidence90
    
    # ROI Settings
    ROI_INTERSECTION_THRESHOLD = 0.2  # Lowered for more lenient intersection0.3
    
    # Tracking Settings
    TRACK_COLORS = {
        'TRACKING': (0, 255, 255),  # Yellow
        'LOCKED': (0, 255, 0)       # Green
    }
    MAX_TRACK_AGE = 3.0  # Maximum time to keep a track (seconds)
    POSITION_THRESHOLD = 100  # Maximum distance for track matching
    MAX_TRACK_AGE = 2.0
    VELOCITY_THRESHOLD = 50.0  # Maximum velocity for track matching
    SMOOTHING_FACTOR = 0.7  # Smoothing factor for Kalman Filter
    MIN_CONFIDENCE_DIFFERENCE = 5.0  # Minimum confidence difference for OCR update
    MIN_TRACK_CONFIDENCE = 0.3  # Minimum confidence to start tracking