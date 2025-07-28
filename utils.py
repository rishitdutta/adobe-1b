import fitz                       # PyMuPDF
from datetime import datetime
from tqdm import tqdm


def parse_pdf(path):
    """
    Returns list[dict]:
        {doc, page, text}
    One element per *page* (simple & fast).
    """
    pages = []
    doc = fitz.open(path)
    for i in range(len(doc)):
        txt = doc.load_page(i).get_text("text")
        if txt.strip():
            pages.append(
                {
                    "doc": path.split("/")[-1],
                    "page": i + 1,          # 1-based
                    "text": txt.strip()
                }
            )
    doc.close()
    return pages


def timestamp():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
