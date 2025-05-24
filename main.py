from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from scipy.spatial.distance import cosine
import os

# Initialize FastAPI app
app = FastAPI(title="Semantic Similarity API", description="API to compute semantic similarity using HF Inference API")

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Fallback to hardcoded token

# Define request body model
class TextPair(BaseModel):
    text1: str
    text2: str

def get_embeddings(texts):
    """
    Fetch embeddings from Hugging Face Inference API.
    Args:
        texts (list): List of texts to get embeddings for
    Returns:
        list: List of embeddings
    """
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": texts})
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=500,
                detail=f"HF API error: Status {response.status_code}, Message: {response.text}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching embeddings: {str(e)}")

# Define the API endpoint
@app.post("/predict", response_model=dict)
async def predict_similarity(text_pair: TextPair):
    """
    Endpoint to compute similarity score between two texts.
    Request body: {"text1": "string", "text2": "string"}
    Response body: {"similarity score": float}
    """
    try:
        # Validate input
        if not text_pair.text1 or not text_pair.text2:
            raise HTTPException(status_code=400, detail="Both text1 and text2 must be non-empty strings")
        
        # Fetch embeddings
        embeddings = get_embeddings([text_pair.text1, text_pair.text2])
        if embeddings is None or len(embeddings) < 2:
            raise HTTPException(status_code=500, detail="Failed to fetch valid embeddings")
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # Compute cosine similarity
        similarity = 1 - cosine(emb1, emb2)
        similarity_score = max(0.0, min(1.0, (similarity + 1) / 2))
        
        # Return response in required format
        return {"similarity score": similarity_score}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "API is running"}
