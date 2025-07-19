
# 541

"""
for the FastAPI app + routing

building the basic framework - search, upload

todo: 1. everything needs a uuid
"""

# imports
from fastapi import File, UploadFile, FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
# from sentence_transformers import CrossEncoder
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import uuid
import numpy as np

from services.vector_index import load_or_create_index, model, add_to_index
from utils.ocr_pipeline import extract_text
# from utils.text_utils import clean_extracted_text
from utils.image_utils import get_image_embeddings
from utils.image_captioning import generate_caption


# configs -----
app = FastAPI()

# RERANKER_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base", use_fast=True)
# RERANKER_MODEL = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

# defining a path for the frontend to display images
app.mount("/memes", StaticFiles(directory="uploaded_memes"), name = "memes")

# loading the crossencoder reranker
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# setting up the routes for upload and search -----

@app.get("/")
def read_root():
    return {"status": "API is running"}
    

@app.post("/upload")
async def upload_files(file: UploadFile = File(...)):
    try:
        # configs
        UPLOAD_DIR = "uploaded_memes"
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # creating unique names for the uploads
        ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # saving to disk
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # text extraction
        extracted_text = extract_text(file_path)
        # cleaned_text = clean_extracted_text(extracted_text)


        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail= "Could not extract text")
        # text_embedding = model.encode([extracted_text]).astype("float32")
        image_embedding = get_image_embeddings(file_path).astype("float32")

        # generating image caption
        image_caption = generate_caption(file_path)

        combined_text = f"{extracted_text.strip()} | {image_caption.strip()}"

        text_embedding = model.encode([combined_text]).astype("float32")

        combined_embedding = np.concatenate([text_embedding, image_embedding], axis=1).astype("float32")

        meta = {
            "filename": filename,
            "extracted_text": extracted_text,
            "caption": image_caption,
            # "url": f"http://127.0.0.1:8001/memes/{filename}" # for the frontend
        }

        add_to_index(combined_embedding, meta)

        return {
            "display_message": "File uploaded!",
            "filename": filename,
            "caption": image_caption,
            "extracted_text": extracted_text
        }
    except Exception as e:
        print("Upload failed", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server failed",
                "detail": str(e)
            }
        )


@app.get("/search")
async def search_memes(q: str = Query(...), k: int = 3):
    k = min(k, 10)

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # vec-embed text
    text_embedding = model.encode([q]).astype("float32")
    placeholder_image = np.zeros((1, 512), dtype="float32")
    query_vec = np.concatenate([text_embedding, placeholder_image], axis=1)

    # loading FAISS, metadata
    index, metadata = load_or_create_index()
    if index.ntotal == 0:
        return {"query": q, "Result": [], "message": "Index is empty. Upload memes."}

    # search
    distances, indices = index.search(query_vec, k)
    min_dist = min(distances[0])
    max_dist = max(distances[0])
    range_dist = max_dist - min_dist if max_dist != min_dist else 1e-5

    # adding the reranking system
    rerank_inputs = []
    valid_indices = []

    for i in indices[0]:
        if i >= len(metadata):
            continue

        meta = metadata[i]
        caption = meta.get("caption", "").strip()
        extracted_text = meta.get("extracted_text", "").strip()

        combined_text = f"{caption} | {extracted_text}".strip()

        # skip empty combined content
        if not combined_text:
            continue

        # rerank_inputs.append((q, combined_text))
        valid_indices.append(i)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(metadata):
            item = metadata[i].copy()
            # item["url"] = f"http://localhost:8001/memes/{item['filename']}"
            similarity_score = 100 * (1 - ((dist - min_dist) / range_dist))
            if similarity_score >= 40:
                item["score"] = float(round(similarity_score, 2))
                results.append(item)


    # preparing the inputs
    # rerank_inputs = [
    #     {"text": q, "text_pair": f"{item['caption']} | {item['extracted_text']}"}
    #     for item in results
    # ]
    # texts = [f"{x['text']} [SEP] {x['text_pair']}" for x in rerank_inputs]
    # tokenized = RERANKER_TOKENIZER(texts, padding=True, truncation=True, return_tensors="pt")

    # # reranking
    # with torch.no_grad():
    #     scores = RERANKER_MODEL(**tokenized).logits.squeeze(-1).tolist()

    # thresholding
    # threshold = 0.4
    # reranked = []
    # for item, score in zip(results, scores):
    #     if score >= threshold:
    #         item["rerank_score"] = round(score, 4)
    #         reranked.append(item)

    # sorting
    # reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    # print ({
    #     "query": q,
    #     "Result": reranked,
    #     "message": "Filtered by reranker with threshold",
    # })
    return {
        "query": q,
        "Result": results,
        "message": "Filtered by reranker with threshold",
    }


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), k: int = 3):
    k = min(k, 10)

    try:
        # saving the uploaded image temporarily
        content = await file.read()
        temp_path = f"temp_{uuid.uuid4().hex[:8]}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        # vec-embed
        image_embedding = get_image_embeddings(temp_path).astype("float32")

        #
        placeholder_text = np.zeros((1, 384), dtype="float32")
        query_vec = np.concatenate([placeholder_text, image_embedding], axis=1)
        # query_vec = image_embedding.astype("float32")

        # loading FAISS and metadata
        index, metadata = load_or_create_index()
        if index.ntotal == 0:
            return {"Result": [], "message": "Index is empty. Upload memes."}

        #
        distances, indices = index.search(query_vec, k)
        min_dist = min(distances[0])
        max_dist = max(distances[0])
        range_dist = max_dist - min_dist if max_dist != min_dist else 1e-5

        # results
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(metadata):
                item = metadata[i].copy()
                # item["url"] = f"http://localhost:8001/memes/{item['filename']}"
                similarity_score = 100 * (1 - ((dist - min_dist) / range_dist))
                if similarity_score >= 40:
                    item["score"] = float(round(similarity_score, 2))
                    results.append(item)

        # clean up temp file
        os.remove(temp_path)

        return {"Result": results}

    except Exception as e:
        print("Image search failed", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )
