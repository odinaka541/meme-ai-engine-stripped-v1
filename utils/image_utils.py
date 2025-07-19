
# 541

"""
image embeddings
"""
import torch
# imports
from PIL import Image
# from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


# init a sentence transformer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# function to vec-embed the images
def get_image_embeddings(image_path: str):
    image = Image.open(image_path).convert("RGB")
    weights = clip_processor(images = image, return_tensors="pt")

    with torch.no_grad():
        outputs = clip_model.get_image_features(**weights)

    embedding = outputs / outputs.norm(dim = -1, keepdim=True) # normalizing

    return embedding.numpy() # 1, 512