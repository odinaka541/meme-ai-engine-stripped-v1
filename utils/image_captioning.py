# 541

# imports
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# configs
PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#
def generate_caption(image_path:str) -> str:
    image = Image.open(image_path)

    inputs = PROCESSOR(image, return_tensors="pt")
    outputs = MODEL.generate(**inputs, max_new_tokens = 100)

    caption = PROCESSOR.decode(outputs[0], skip_special_tokens = True)

    return caption
