"""
Configuration for your CLI project

🔧 Customize your CLI by editing these values:
1. Change PROJECT_INFO to match your project
2. Modify DEFAULT_VALUES as needed
3. Add your own configuration constants below
"""

PROJECT_INFO = {
    "name": "OCR Performance Test",
    "version": "0.1.0", 
    "description": "OCR models performance testing CLI tool"
}

DEFAULT_VALUES = {
    "output_format": "json",
    "log_level": "INFO", 
    "max_workers": 4,
    "confidence_threshold": 0.5,
    "save_outputs": True
}

OCR_SETTINGS = {
    "models": ["easyocr", "paddleocr", "pytesseract"],
    "languages": ["en", "th"],
    "output_dir": "outputs",
    "benchmark_iterations": 3
}

APP_NAME = "ocr-performance-test"