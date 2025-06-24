from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # ringan dan bagus

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbeddingRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)