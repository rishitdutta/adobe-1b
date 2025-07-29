"""
Load every model once, CPU-only, fully offline.
Total RAM footprint ≈ 450 MB.
"""
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_models():
    device = "cpu"          # mandatory for challenge (no-GPU)

    # 1) Summariser & QA generator
    sum_name = "google/flan-t5-small"       # ~242 MB
    sum_tok  = AutoTokenizer.from_pretrained(sum_name)
    sum_mod  = AutoModelForSeq2SeqLM.from_pretrained(sum_name).to(device)

    # 2) Embedder
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", # ~90MB
        device=device
    )

    # 3) Reranker (answer-aware)
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", # ~80 MB, fast and small
        device=device,
        max_length=512
    )

    return {
        "sum_tok": sum_tok,
        "sum_mod": sum_mod,
        "embedder": embedder,
        "reranker": reranker,
        "device": device,
    }
