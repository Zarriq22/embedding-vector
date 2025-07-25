from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import uvicorn
import re

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # ringan dan bagus

factory = StemmerFactory()
stemmer = factory.create_stemmer()

class EmbeddingRequest(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    # 1. Case folding
    text = text.lower()
    # 2. Remove karakter non-alfabet (opsional, bisa disesuaikan)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # 3. Stemming
    text = stemmer.stem(text)
    return text

@app.post("/embed")
async def embed_text(request: EmbeddingRequest):
    # Preprocess sebelum di-encode
    processed_text = preprocess_text(request.text)

    # Generate embedding
    embedding = model.encode(processed_text).tolist()
    return {"embedding": embedding}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)