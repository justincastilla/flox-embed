import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CPU by default: this model is small enough not to need an accelerator, and it
# keeps results identical everywhere. Set EMBED_DEVICE=mps/cuda to override.
DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Downloads once into the Flox env cache, then loads from disk forever after.
    state["model"] = SentenceTransformer(MODEL_ID, device=DEVICE)
    state["dims"] = state["model"].get_sentence_embedding_dimension()
    yield
    state.clear()


app = FastAPI(title="flox-embed", lifespan=lifespan)


class EmbedRequest(BaseModel):
    input: list[str] = Field(..., min_length=1, max_length=256)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "dimensions": state.get("dims"),
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    vectors = state["model"].encode(req.input).tolist()
    return {
        "model": MODEL_ID,
        "dimensions": len(vectors[0]),
        "embeddings": vectors,
    }
