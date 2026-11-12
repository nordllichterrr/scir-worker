from pathlib import Path
import os

# Reduce Hugging Face Hub warnings when HF_TOKEN is not set (optional for higher rate limits)
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = ROOT / "results"


for p in (PROCESSED_DIR, EMBEDDINGS_DIR, INDEX_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


FAISS_INDEX_PATH = INDEX_DIR / "items.faiss"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "item_embeddings.npy"
EMBEDDING_MAP = EMBEDDINGS_DIR / "id2idx.json"


CANDIDATE_POOL = 500
RERANK_TOPK = 20

RANDOM_SEED = 42

LLM_BACKEND = "hf" 

HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


HF_MAX_NEW_TOKENS = 256
HF_TEMPERATURE = 0.3
HF_TOP_P = 0.9
HF_DEVICE = "cpu"  


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

