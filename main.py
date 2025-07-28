"""
End-to-end pipeline

ENV variables expected by the Docker wrapper:
    PDF_DIR   – folder containing pdfs
    PERSONA   – persona string
    JOB       – job-to-be-done string
    OUT_FILE  – (optional) where to write JSON (default: stdout only)
"""
import os, json, time
import numpy as np
from utils import parse_pdf, timestamp
from models import load_models
import faiss
from tqdm import tqdm
from transformers import GenerationConfig

# ---------------- config -----------------
TOP_K     = 30        # candidates from FAISS
TOP_N_OUT = 5         # final passages
GEN_CFG = GenerationConfig(
    max_length=128,
    temperature=0.7,
    top_p=0.95,
    do_sample=False,
    num_beams=4,
    early_stopping=True,
)
# -----------------------------------------


def summarise(long_text, mdl):
    tok, mod = mdl["sum_tok"], mdl["sum_mod"]
    chunks = [long_text[i:i + 2048] for i in range(0, len(long_text), 2048)]
    partial = []
    for c in chunks:
        ids = tok("summarize: " + c, return_tensors="pt", truncation=True).to(mdl["device"])
        out = mod.generate(**ids, **GEN_CFG.to_dict())
        partial.append(tok.decode(out[0], skip_special_tokens=True))

    combined = " ".join(partial)
    ids = tok("summarize: " + combined, return_tensors="pt").to(mdl["device"])
    final = mod.generate(**ids, **GEN_CFG.to_dict())
    return tok.decode(final[0], skip_special_tokens=True)


def generate_answer(persona, job, context_sum, mdl):
    prompt = (
        f"Persona: {persona}\nJob: {job}\nContext: {context_sum}\n\n"
        "Provide a concise answer the persona would find most useful."
    )
    tok, mod = mdl["sum_tok"], mdl["sum_mod"]
    ids = tok(prompt, return_tensors="pt", truncation=True).to(mdl["device"])
    out = mod.generate(**ids, **GEN_CFG.to_dict())
    return tok.decode(out[0], skip_special_tokens=True)


def build_faiss(vectors):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def retrieve(pages, answer, mdl):
    embedder = mdl["embedder"]
    # encode pages
    page_texts = [p["text"] for p in pages]
    p_emb = embedder.encode(page_texts, batch_size=64, show_progress_bar=True)
    # encode query
    q_emb = embedder.encode(answer, convert_to_tensor=False)
    # build & search
    idx = build_faiss(np.array(p_emb, dtype="float32"))
    faiss.normalize_L2(q_emb.reshape(1, -1))
    _, I = idx.search(q_emb.reshape(1, -1), TOP_K)
    return [pages[i] for i in I[0]]


def rerank(cands, answer, mdl):
    pairs = [[answer, c["text"]] for c in cands]
    scores = mdl["reranker"].predict(pairs, batch_size=8, show_progress_bar=True)
    ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
    return ranked[:TOP_N_OUT]


def section_title(text):
    # crude title: first non-empty line ≤ 12 words
    for line in text.splitlines():
        l = line.strip()
        if l:
            return " ".join(l.split()[:12])
    return "Untitled section"


def pipeline(pdf_paths, persona, job):
    t0 = time.time()
    mdl = load_models()

    # ------------ ingest ------------
    pages = []
    for p in pdf_paths:
        pages.extend(parse_pdf(p))

    # ------------ summarise corpus ------------
    corpus_text = " ".join(p["text"] for p in pages)
    summary = summarise(corpus_text, mdl)

    # ------------ answer scent ------------
    answer = generate_answer(persona, job, summary, mdl)

    # ------------ retrieve & rerank ------------
    candidates = retrieve(pages, answer, mdl)
    top_passages = rerank(candidates, answer, mdl)

    # ------------ build JSON ------------
    output = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp(),
            "latency_sec": round(time.time() - t0, 2),
        },
        "extracted_sections": [],
        "subsection_analysis": [],
    }

    for rank, (score, passage) in enumerate(top_passages, 1):
        # Section / page entry
        output["extracted_sections"].append(
            {
                "document": passage["doc"],
                "page_number": passage["page"],
                "section_title": section_title(passage["text"]),
                "importance_rank": rank,
            }
        )
        # Sub-section analysis (refined text)
        output["subsection_analysis"].append(
            {
                "document": passage["doc"],
                "refined_text": passage["text"][:1000],   # truncated raw text
                "page_number": passage["page"],
            }
        )

    return output


# ---------------- CLI entry ----------------
if __name__ == "__main__":
    pdf_dir = os.getenv("PDF_DIR", "documents")
    persona = os.getenv("PERSONA", "Generic user")
    job     = os.getenv("JOB", "Generic task")
    out_f   = os.getenv("OUT_FILE")          # optional

    pdfs = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    result_json = pipeline(pdfs, persona, job)

    # 1) stdout
    print(json.dumps(result_json, indent=2, ensure_ascii=False))

    # 2) optional file
    if out_f:
        with open(out_f, "w", encoding="utf-8") as fh:
            json.dump(result_json, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_f}")
