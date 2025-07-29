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
from summarizer import Summarizer

# ---------------- config -----------------
TOP_K     = 30        # candidates from FAISS
TOP_N_OUT = 5         # final passages
GEN_CFG = GenerationConfig(
    max_length=128,
    do_sample=False,
    num_beams=4,
    early_stopping=True,
)
PARAPHRASE_CFG = GenerationConfig(
    max_length=80,  # Slightly longer to allow natural phrasing
    num_beams=4,
    do_sample=True,  # Enables variety for casual tone
    repetition_penalty=1.2,  # Penalizes loops like "lyly.comlyly.com"
    early_stopping=True,
)
# -----------------------------------------

""" def summarise(long_text, mdl, gen_cfg):
    tok, mod = mdl["sum_tok"], mdl["sum_mod"]
    chunks = [long_text[i:i + 2048] for i in range(0, len(long_text), 2048)]
    partial = []
    for c in tqdm(chunks, desc="Summarising chunks"):
        ids = tok("summarize: " + c, return_tensors="pt", truncation=True).to(mdl["device"])
        out = mod.generate(**ids, **gen_cfg)
        partial.append(tok.decode(out[0], skip_special_tokens=True))

    if not partial:
        return ""

    combined = " ".join(partial)
    ids = tok("summarize: " + combined, return_tensors="pt", truncation=True).to(mdl["device"])
    final = mod.generate(**ids, **gen_cfg)
    return tok.decode(final[0], skip_special_tokens=True) """

def summarise_extractive(long_text, mdl, num_sentences=10):
    # Uses the embedding model you already loaded for sentence vectors
    model = Summarizer(model=mdl["embedder"])
    summary = model(long_text, num_sentences=num_sentences)
    return summary

def generate_answer(persona, job, context_sum, mdl, gen_cfg):
    prompt = (
        f"Persona: {persona}\nJob: {job}\nContext: {context_sum}\n\n"
        "Provide a concise answer the persona would find most useful."
    )
    tok, mod = mdl["sum_tok"], mdl["sum_mod"]
    ids = tok(prompt, return_tensors="pt", truncation=True).to(mdl["device"])
    out = mod.generate(**ids, **gen_cfg)
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

def pipeline(pdf_paths, persona, job):
    t0 = time.time()
    mdl = load_models()

    # ------------ ingest ------------
    pages = []
    for p in pdf_paths:
        pages.extend(parse_pdf(p))

    # ------------ set generation config for summarizer ------------
    # T5 needs decoder_start_token_id, which is pad_token_id
    sum_gen_cfg = GEN_CFG.to_dict()
    sum_gen_cfg["decoder_start_token_id"] = mdl["sum_mod"].config.pad_token_id

    # ------------ summarise corpus ------------
    corpus_text = " ".join(p["text"] for p in pages)
    summary = summarise_extractive(corpus_text, mdl, num_sentences=15)

    # ------------ answer scent ------------
    answer = generate_answer(persona, job, summary, mdl, sum_gen_cfg)

    # ------------ retrieve & rerank ------------
    candidates = retrieve(pages, answer, mdl)
    top_passages = rerank(candidates, answer, mdl)

    # ------------ build JSON ------------
    output = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
        },
        "extracted_sections": [],
        "subsection_analysis": [],
    }

    for rank, (score, passage) in enumerate(top_passages, 1):
        # Section / page entry
        output["extracted_sections"].append(
            {
                "document": os.path.basename(passage["doc"]),
                "section_title": passage["title"],
                "importance_rank": rank,
                "page_number": passage["page"],
            }
        )
        # Sub-section analysis (refined text)
        output["subsection_analysis"].append(
            {
                "document": os.path.basename(passage["doc"]),
                "refined_text": passage["text"][:1000],   # truncated raw text
                "page_number": passage["page"],
            }
        )

    return output

def paraphrase(text, mdl):
    tok, mod = mdl["sum_tok"], mdl["sum_mod"]
    
    # [FIX] Input validation (optional but recommended to avoid edge-case crashes)
    if not isinstance(text, str) or not text.strip():
        return text  # Return original if invalid/empty
    
    # [FIX] Use a more explicit prompt for better Flan-T5 results (optional improvement)
    prompt = f"Paraphrase the following: {text}"
    
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(mdl["device"])
    
    # [FIX] Create a dict from the config, add decoder_start_token_id to it (avoids duplicate kwargs)
    para_cfg_dict = PARAPHRASE_CFG.to_dict()
    para_cfg_dict["decoder_start_token_id"] = mod.config.pad_token_id
    
    out = mod.generate(
        **ids,
        **para_cfg_dict  # Unpack only once, with everything in the dict
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()

# ---------------- CLI entry ----------------
if __name__ == "__main__":
    input_json_path = os.getenv("INPUT_JSON", "challenge1b_input.json")
    pdf_dir = os.getenv("PDF_DIR", "documents")
    out_f = os.getenv("OUT_FILE", "challenge1b_output.json")

    # 1) Load configuration from input JSON
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        raw_persona = input_data["persona"]["role"]
        raw_job = input_data["job_to_be_done"]["task"]
        documents = input_data["documents"]
        pdfs = [os.path.join(pdf_dir, doc["filename"]) for doc in documents]
    except (FileNotFoundError, KeyError) as e:
        raise SystemExit(f"Error reading input JSON '{input_json_path}': {e}")

    if not pdfs:
        raise SystemExit(f"No PDF files listed in input JSON found in '{pdf_dir}'")

    # 2) load models once, then canonicalize the persona/job
    mdl = load_models()
    persona = paraphrase(raw_persona, mdl)
    job     = paraphrase(raw_job, mdl)

    result_json = pipeline(pdfs, persona, job)

    # 3) stdout
    print(json.dumps(result_json, indent=2, ensure_ascii=False))

    # 4) optional file
    if out_f:
        with open(out_f, "w", encoding="utf-8") as fh:
            json.dump(result_json, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_f}")