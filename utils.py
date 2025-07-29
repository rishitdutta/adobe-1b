import fitz  # PyMuPDF
import pdfplumber
import re
from datetime import datetime, timezone
from collections import defaultdict


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def parse_pdf(pdf_path):
    """
    Extracts structured text chunks from a PDF.
    - Uses pdfplumber to identify headings based on font size/weight.
    - Uses PyMuPDF (fitz) to extract text blocks under those headings.
    - Returns a list of passages, each with its text and inferred title.
    """
    print(f"Processing: {pdf_path}")
    # 1. Extract a structured outline with pdfplumber
    lines = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                char_lines = defaultdict(list)
                for char in page.chars:
                    # Round 'top' to group characters on the same line
                    y = round(char['top'], 1)
                    char_lines[y].append(char)

                for y, chars in sorted(char_lines.items()):
                    sorted_chars = sorted(chars, key=lambda c: c['x0'])
                    raw_text = ''.join(c['text'] for c in sorted_chars).strip()
                    
                    if not raw_text:
                        continue

                    font_sizes = [c.get('size') for c in sorted_chars if isinstance(c.get('size'), (int, float))]
                    font_names = [c.get('fontname', '') for c in sorted_chars]
                    bold_count = sum(1 for name in font_names if 'Bold' in name or 'Black' in name)
                    
                    lines.append({
                        'text': raw_text,
                        'fontsize': round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 0,
                        'bold': (bold_count / len(font_names)) >= 0.7 if font_names else False,
                        'page': page.page_number,
                    })
    except Exception as e:
        print(f"  Warning: pdfplumber failed on {pdf_path}: {e}. Falling back to basic text extraction.")
        # Fallback to simple text extraction if structure parsing fails
        doc = fitz.open(pdf_path)
        return [{"doc": pdf_path, "page": p.number + 1, "title": f"Page {p.number + 1}", "text": p.get_text("text")} for p in doc]


    # Identify potential headings (bold and larger than average font size)
    font_sizes = [l['fontsize'] for l in lines if l['text']]
    avg_fs = sum(font_sizes) / len(font_sizes) if font_sizes else 10
    
    headings = []
    for line in lines:
        # A heading is bold and has a larger-than-average font size.
        if line['bold'] and line['fontsize'] > avg_fs:
            headings.append({
                "text": line['text'],
                "page": line['page'],
            })

    # 2. Use PyMuPDF to get text blocks and associate them with headings
    doc = fitz.open(pdf_path)
    passages = []
    current_heading = "Introduction"

    for page_num, page in enumerate(doc, 1):
        page_headings = [h['text'] for h in headings if h['page'] == page_num]
        
        blocks = page.get_text("blocks")
        for block in blocks:
            block_text = block[4].strip().replace("\n", " ")
            if not block_text:
                continue

            # If this block's text is a heading, update the current heading
            if block_text in page_headings:
                current_heading = block_text
            else:
                passages.append({
                    "doc": pdf_path,
                    "page": page_num,
                    "title": current_heading,
                    "text": block_text
                })
    
    # If no passages were created, fall back to one passage per page
    if not passages:
        return [{"doc": pdf_path, "page": p.number + 1, "title": f"Page {p.number + 1}", "text": p.get_text("text")} for p in doc]

    return passages
