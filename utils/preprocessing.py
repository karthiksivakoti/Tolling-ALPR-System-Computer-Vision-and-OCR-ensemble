# utils/preprocessing.py
import cv2
import numpy as np

class ImagePreprocessor:
    @staticmethod
    def preprocess_plate(plate_img, target_size=(300, 100)):
        """Preprocess license plate image for OCR"""
        try:
            # Resize
            plate_img = cv2.resize(plate_img, target_size)
            
            # Convert to grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Remove noise
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
            
    @staticmethod
    def enhance_plate(plate_img):
        """Enhance plate image quality"""
        try:
            # Increase contrast
            lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            print(f"Error in enhancement: {str(e)}")
            return plate_img