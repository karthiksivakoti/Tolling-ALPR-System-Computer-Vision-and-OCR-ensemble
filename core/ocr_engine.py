# core/ocr_engine.py
import pytesseract
import easyocr
import cv2
import numpy as np
from config.settings import Config
import time
import os
from typing import Tuple, Optional

class OCREngine:
    def __init__(self):
        try:
            # Initialize both OCR engines
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
            self.tesseract_config = f'--psm 7 -c tessedit_char_whitelist={Config.ALLOWED_CHARS}'
            self.easyocr_reader = easyocr.Reader(['en'])
            
            # Create debug directory
            os.makedirs("debug_plates", exist_ok=True)
            print("OCR Engine initialized successfully")
            
        except Exception as e:
            print(f"Error initializing OCR engine: {str(e)}")
            raise e

    def process_plate(self, plate_img: np.ndarray, track_id: int) -> Tuple[str, float]:
        """Process with both OCR engines and return best result"""
        try:
            if plate_img is None:
                return "", 0

            # Basic preprocessing
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Try both OCR engines
            text1, conf1 = self._tesseract_ocr(thresh)
            text2, conf2 = self._easyocr_ocr(thresh)

            # Choose result with higher confidence
            if conf1 > conf2:
                cleaned_text = self._clean_text(text1)
                return cleaned_text, conf1
            else:
                cleaned_text = self._clean_text(text2)
                return cleaned_text, conf2

        except Exception as e:
            print(f"Process plate error: {str(e)}")
            return "", 0

    def _tesseract_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Process image with Tesseract OCR"""
        try:
            result = pytesseract.image_to_data(
                image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            valid_words = []
            total_conf = 0
            count = 0
            
            for i, conf in enumerate(result['conf']):
                if conf > 0:  # Valid detection
                    valid_words.append(result['text'][i])
                    total_conf += conf
                    count += 1
            
            text = "".join(valid_words)
            confidence = total_conf / count if count > 0 else 0
            
            return text, confidence
            
        except Exception as e:
            print(f"Tesseract error: {str(e)}")
            return "", 0

    def _easyocr_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Process image with EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image)
            
            if results:
                text = "".join([result[1] for result in results])
                confidence = sum(result[2] for result in results) / len(results) * 100
                return text, confidence
            
            return "", 0
            
        except Exception as e:
            print(f"EasyOCR error: {str(e)}")
            return "", 0

    def _clean_text(self, text: str) -> str:
        """Clean and standardize OCR text"""
        # Remove any non-alphanumeric characters and convert to uppercase
        cleaned = ''.join(c for c in text if c.isalnum()).upper()
        
        # Common OCR mistake corrections
        replacements = {
            '0': 'O',  # Replace zero with letter O
            '1': 'I',  # Replace one with letter I
            '5': 'S',  # Replace 5 with S
            '8': 'B',  # Replace 8 with B
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned