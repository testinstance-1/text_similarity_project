import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = "your_huggingface_api_key"  # Set via environment variable in Render

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_embeddings(texts):
    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    response.raise_for_status()
    return np.array(response.json())

def compute_similarity(text1: str, text2: str) -> float:
    embeddings = get_embeddings([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(similarity, 0, 1))
