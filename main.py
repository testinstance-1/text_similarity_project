from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from scipy.spatial.distance import cosine
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Fetch similarity scores from Hugging Face Inference API.
    Args:
        texts (list): List of two texts (source and comparison)
    Returns:
        list: List of similarity scores
    """
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": {"source_sentence": texts[0], "sentences": [texts[1]]}}
    try:
        logger.info(f"Sending request to HF API with payload: {payload}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        logger.info(f"HF API response status: {response.status_code}, content: {response.text[:200]}")
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = {
                "status_code": response.status_code,
                "response_text": response.text,
                "headers": dict(response.headers)
            }
            logger.error(f"HF API error: {error_detail}")
            raise HTTPException(status_code=500, detail=f"HF API error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching embeddings: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
        
        # Fetch similarity scores
        scores = get_embeddings([text_pair.text1, text_pair.text2])
        if not isinstance(scores, list) or len(scores) < 1:
            raise HTTPException(status_code=500, detail="Invalid embeddings format")
        # Extract first similarity score
        similarity_score = scores[0]
        if not isinstance(similarity_score, (int, float)):
            raise HTTPException(status_code=500, detail="Invalid similarity score format")
        
        # Normalize score to [0,1] (handles negative scores)
        normalized_score = (similarity_score + 1) / 2 if similarity_score < 0 else similarity_score
        normalized_score = max(0.0, min(1.0, float(normalized_score)))
        
        # Return response in required format
        return {"similarity score": normalized_score}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "API is running"}

# Debug endpoint to check token
@app.get("/debug")
async def debug_token():
    """
    Debug endpoint to verify token configuration.
    """
    return {"api_token_configured": bool(API_TOKEN), "token_preview": API_TOKEN[:5] + "..." if API_TOKEN else "None"}
