from fastapi import FastAPI
from pydantic import BaseModel
from hf_model import compute_similarity

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/predict")
def predict(pair: TextPair):
    score = compute_similarity(pair.text1, pair.text2)
    return {"similarity score": round(score, 3)}
