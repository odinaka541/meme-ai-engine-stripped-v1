
# 541

"""
helper script to batch process the text-extraction
"""

# imports -----
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract, cv2
import numpy as np

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# configs -----
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# force pytesseract to decode stdout as utf-8 instead of 'charmap'
import pytesseract
import subprocess

# patch subprocess.run inside pytesseract to force utf-8 decoding
original_subprocess_run = subprocess.run
def patched_subprocess_run(*args, **kwargs):
    kwargs["encoding"] = "utf-8"
    return original_subprocess_run(*args, **kwargs)
# apply patch only once
if not hasattr(pytesseract.pytesseract, "_subprocess_patched"):
    subprocess.run = patched_subprocess_run
    pytesseract.pytesseract._subprocess_patched = True


# simple, lightweight image-preprocesing
def preprocess_image(image_path: str):
    img = Image.open(image_path).convert("L")  # grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.filter(ImageFilter.SHARPEN)
    np_img = np.array(img)
    blur = cv2.bilateralFilter(np_img, 9, 75, 75)
    return Image.fromarray(blur)

def extract_text(image_path:str):
    img = preprocess_image(image_path)
    custom_config = r'--oem 1 --psm 6'
    try:
        return pytesseract.image_to_string(img, config=custom_config)
    except Exception as e:
        print(f"Unicode decode error: {e}")
        return ""