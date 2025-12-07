"""
OCR processor for blood reports
Author: Shaurya Parshad
Description: Extracts medical indicators from blood reports using OCR
"""

import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re
import os
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Feature patterns to look for in blood reports
INDICATOR_PATTERNS = {
    'hemo': r'(?:hemoglobin|hb|hgb)[:\s]*([0-9.]+)',
    'pcv': r'(?:pcv|packed cell volume|hematocrit|hct)[:\s]*([0-9.]+)',
    'sg': r'(?:specific gravity|sg)[:\s]*([0-9.]+)',
    'gfr': r'(?:gfr|egfr|glomerular filtration rate)[:\s]*([0-9.]+)',
    'rbcc': r'(?:rbc|red blood cell|rbcc|erythrocyte)[:\s]*([0-9.]+)',
    'al': r'(?:albumin|alb)[:\s]*([0-9]+)',
    'dm': r'(?:diabetes|diabetic|glucose)[:\s]*(yes|no|positive|negative|present|absent|elevated|normal)',
    'htn': r'(?:hypertension|htn|blood pressure|bp)[:\s]*(yes|no|positive|negative|elevated|normal|high)',
    'sod': r'(?:sodium|na)[:\s]*([0-9.]+)',
    'bp': r'(?:blood pressure|bp|systolic)[:\s]*([0-9.]+)',
    'sc': r'(?:serum creatinine|creatinine|creat|crea)[:\s]*([0-9.]+)'
}


def preprocess_image(image_path):
    """
    Preprocess image for better OCR accuracy

    Args:
        image_path: Path to image file

    Returns:
        numpy array: Preprocessed image
    """
    try:
        # Read image
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)

        return denoised

    except Exception as e:
        print(f"Image preprocessing error: {e}")
        # Return original if preprocessing fails
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def extract_text_from_image(image_path):
    """
    Extract text from image using Tesseract OCR

    Args:
        image_path: Path to image file

    Returns:
        str: Extracted text
    """
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)

        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6'

        # Extract text
        text = pytesseract.image_to_string(processed_img, config=custom_config)

        return text.lower()

    except Exception as e:
        print(f"Image OCR error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file

    Args:
        pdf_path: Path to PDF file

    Returns:
        str: Extracted text from all pages
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)

        full_text = ""
        for i, image in enumerate(images):
            print(f"Processing PDF page {i + 1}/{len(images)}")

            # Extract text from each page
            text = pytesseract.image_to_string(image)
            full_text += text.lower() + "\n"

        return full_text

    except Exception as e:
        print(f"PDF OCR error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def parse_indicators(text):
    """
    Parse medical indicators from extracted text

    Args:
        text: Extracted text from OCR

    Returns:
        dict: Parsed indicators
    """
    indicators = {}

    for indicator, pattern in INDICATOR_PATTERNS.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            value = match.group(1)

            # Convert to appropriate type
            if indicator in ['dm', 'htn']:
                # Binary indicators
                positive_terms = ['yes', 'positive', 'present', 'elevated', 'high']
                indicators[indicator] = 1 if any(term in value.lower() for term in positive_terms) else 0
            else:
                # Numeric indicators
                try:
                    indicators[indicator] = float(value)
                    break  # Use first valid match
                except ValueError:
                    continue

    return indicators


def validate_indicators(indicators):
    """
    Validate extracted indicators against reasonable ranges

    Args:
        indicators: Dictionary of extracted indicators

    Returns:
        tuple: (validated_indicators, warnings)
    """
    # Reasonable ranges for each indicator
    ranges = {
        'hemo': (3, 20),
        'pcv': (15, 60),
        'sg': (1.000, 1.030),
        'gfr': (5, 150),
        'rbcc': (1.5, 7),
        'al': (0, 5),
        'dm': (0, 1),
        'htn': (0, 1),
        'sod': (120, 160),
        'bp': (80, 200),
        'sc': (0.3, 12)
    }

    validated = {}
    warnings = []

    for key, value in indicators.items():
        if key in ranges:
            min_val, max_val = ranges[key]
            if min_val <= value <= max_val:
                validated[key] = value
            else:
                warnings.append(
                    f"{key.upper()}: {value} is outside expected range "
                    f"({min_val}-{max_val}). Please verify."
                )
                # Still include it but flag it
                validated[key] = value

    return validated, warnings


def process_blood_report(file_path):
    """
    Main function to process blood report and extract indicators

    Args:
        file_path: Path to uploaded blood report file

    Returns:
        dict: Processing result with extracted indicators
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': 'File not found'
            }

        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()

        print(f"Processing file: {file_path}")
        print(f"File type: {file_ext}")

        # Extract text based on file type
        if file_ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            text = extract_text_from_image(file_path)
        else:
            return {
                'success': False,
                'error': f'Unsupported file format: {file_ext}. Please upload PDF, JPG, or PNG.'
            }

        if not text:
            return {
                'success': False,
                'error': 'Could not extract text from file. Please ensure the image is clear and text is readable.'
            }

        print(f"Extracted text length: {len(text)} characters")

        # Parse indicators
        indicators = parse_indicators(text)

        if not indicators:
            return {
                'success': False,
                'error': 'Could not find any medical indicators in the report. Please upload a valid blood test report.',
                'raw_text': text[:500]  # Return first 500 chars for debugging
            }

        print(f"Found indicators: {list(indicators.keys())}")

        # Validate
        validated, warnings = validate_indicators(indicators)

        return {
            'success': True,
            'indicators': validated,
            'warnings': warnings,
            'raw_text': text[:500],  # First 500 chars for reference
            'extracted_count': len(validated)
        }

    except Exception as e:
        print(f"Blood report processing error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'Processing error: {str(e)}'
        }


def get_missing_indicators(extracted_indicators):
    """
    Identify which required indicators are missing

    Args:
        extracted_indicators: Dictionary of extracted indicators

    Returns:
        list: List of missing indicator names
    """
    required = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']
    missing = [ind for ind in required if ind not in extracted_indicators]
    return missing