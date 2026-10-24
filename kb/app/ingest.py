"""Ingestion: PDF -> pages (with OCR fallback) -> page-tracked chunks -> auto-built document cards (metadata layer)."""
import os, re, json, glob, shutil, zipfile, pickle, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from . import config as C

# ---------------- page extraction ----------------
def _pages_pymupdf(path):
    import fitz
    d = fitz.open(path); out = [p.get_text("text") for p in d]; d.close(); return out

def _pages_pdftotext(path):
    r = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, timeout=900)
    return r.stdout.decode("utf8", "ignore").split("\f")[:-1] or [""]

def _pages_plumber(path):
    import pdfplumber
    with pdfplumber.open(path) as p: return [(pg.extract_text() or "") for pg in p.pages]

def _ocr_page(path, pno):
    """OCR a single 1-indexed page (scanned PDFs, e.g. SAMA file)."""
    try:
        import pytesseract
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["pdftoppm", "-r", "200", "-f", str(pno), "-l", str(pno), "-png", path, f"{td}/p"],
                           capture_output=True, timeout=120)
            f = sorted(glob.glob(f"{td}/p*.png"))
            return pytesseract.image_to_string(Image.open(f[0])) if f else ""
    except Exception:
        try:
            import fitz, pytesseract, io
            from PIL import Image
            d = fitz.open(path); pix = d[pno - 1].get_pixmap(dpi=200)
            return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
        except Exception: return ""

def extract_pages(path):
    pages = None
    for fn in (_pages_pymupdf, _pages_pdftotext, _pages_plumber):
        try: pages = fn(path); break
        except Exception: continue
    if pages is None: return []
    weak = [i for i, t in enumerate(pages) if len(t.strip()) < C.OCR_MIN_CHARS]
    if weak and len(weak) <= 400:      # OCR only the image-only pages
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            for i, t in zip(weak, ex.map(lambda i: _ocr_page(path, i + 1), weak)):
                if t.strip(): pages[i] = t
    return pages

# ---------------- chunking (page-tracked, structure-aware) ----------------
_HEAD = re.compile(r"^(?:[A-Z][A-Z0-9 ,&/\-\(\):.]{4,90}|(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+[A-Z][^\n]{3,90})$")

def _clean(t):
    t = t.replace("\x00", " ").replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    return t

def chunk_page(text, doc, page, max_chars):
    text = _clean(text); out = []; section = ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 2 and len(text) > max_chars:                       # no blank lines -> group lines
        lines = [l for l in text.split("\n") if l.strip()]; paras, cur = [], ""
        for l in lines:
            if len(cur) + len(l) > max_chars * 0.8 and cur: paras.append(cur); cur = l
            else: cur = (cur + " " + l).strip()
        if cur: paras.append(cur)
    buf = ""
    def flush():
        nonlocal buf
        if len(buf.strip()) >= 25:
            out.append({"doc": doc, "page": page, "section": section, "text": buf.strip()})
        buf = ""
    for p in paras:
        first = p.split("\n")[0].strip()
        if len(p) < 100 and _HEAD.match(first): flush(); section = first[:100]; buf = p; continue
        if len(p) > max_chars * 1.5:                                    # split very long paragraph on sentences
            sents = re.split(r"(?<=[.;:])\s+", p)
            for s in sents:
                if len(buf) + len(s) > max_chars and buf: flush()
                buf += (" " if buf else "") + s
            continue
        if len(buf) + len(p) > max_chars and buf: flush()
        buf += ("\n" if buf else "") + p
    flush()
    return out

# ---------------- document cards (automatic metadata layer) ----------------
YEAR = re.compile(r"\b(?:FY\s?)?(20\d\d)(?:\s?[-–/]\s?(\d{2,4}))?\b")

CARD_SYS = ("You build metadata cards for documents in a financial/regulatory knowledge base. "
            "Use only the text given. Respond only with JSON.")
CARD_USER = """Filename: {fn}
Pages: {n}
First pages text:
\"\"\"{head}\"\"\"

Return JSON: {{"title": str, "entity": "company/regulator/author that issued or is the subject (or '')",
"doc_type": "annual report | regulation/proposal | supervisory guidance | accounting standard | research paper | handbook | memo | fact sheet | other",
"period": "reporting year / fiscal year / publication year as written, e.g. 'FY2025-26' or '2023'",
"jurisdiction": "e.g. US, India, Saudi Arabia, Basel/global, or ''",
"topics": [up to 10 short topic keywords],
"summary": "one sentence describing what the document covers and what distinguishes it from similar documents"}}"""

def heuristic_card(fn, pages):
    head = " ".join(pages[:3])[:4000]
    full_search = f"{fn} {head}"
    yrs = sorted({m.group(0) for m in YEAR.finditer(full_search[:2000])})[:4]
    
    entity = ""
    clean_base = re.sub(r"[_\-]+", " ", os.path.splitext(fn)[0])
    
    entity_patterns = [
        (r"(?i)\bGammon\s+India\b", "Gammon India Limited"),
        (r"(?i)\bBhadora\s+Industries\b", "Bhadora Industries Limited"),
        (r"(?i)\bCEAT\b", "CEAT Limited"),
        (r"(?i)\bGRM\s+Overseas\b", "GRM Overseas Limited"),
        (r"(?i)\bInox\s+Wind\b", "Inox Wind Limited"),
        (r"(?i)\bMetal\s+Coatings\b", "Metal Coatings (India) Limited"),
        (r"(?i)\bRefex\s+Industries\b|\bRefex\b", "Refex Industries Limited"),
        (r"(?i)\bAldar\b", "Aldar Properties PJSC"),
        (r"(?i)\bInfosys\b", "Infosys Limited"),
        (r"(?i)\bBSF\b|\bBanque\s+Saudi\s+Fransi\b", "Banque Saudi Fransi (BSF)"),
        (r"(?i)\bSAMA\b|\bSaudi\s+Central\s+Bank\b", "Saudi Central Bank (SAMA)"),
        (r"(?i)\bFDIC\b|\bFederal\s+Deposit\s+Insurance\b", "Federal Deposit Insurance Corporation (FDIC)"),
        (r"(?i)\bOCC\b|\bComptroller\s+of\s+the\s+Currency\b", "Office of the Comptroller of the Currency (OCC)"),
        (r"(?i)\bFASB\b|\bAccounting\s+Standards\s+Codification\b", "Financial Accounting Standards Board (FASB)"),
        (r"(?i)\bBCBS\b|\bBasel\s+Committee\b", "Basel Committee on Banking Supervision (BCBS)"),
        (r"(?i)\bFederal\s+Reserve\b|\bBoard\s+of\s+Governors\b", "Board of Governors of the Federal Reserve System"),
        (r"(?i)\bRiskMetrics\b", "J.P. Morgan / Reuters"),
        (r"(?i)\bErnst\s*&\s*Young\b|\bEY\b", "Ernst & Young (EY)"),
        (r"(?i)\bHeston\b", "Steven L. Heston")
    ]
    for pat, name in entity_patterns:
        if re.search(pat, full_search):
            entity = name
            break
            
    if not entity:
        m = re.search(r"^([A-Za-z0-9\s&]+?)(?:_Annual Report| Annual Report|\s+Integrated|\s+FY\d)", os.path.splitext(fn)[0])
        if m:
            entity = m.group(1).strip()
            
    jurisdiction = ""
    if re.search(r"(?i)\b(India|SEBI|crore|lakhs|rupee|INR)\b", full_search):
        jurisdiction = "India"
    elif re.search(r"(?i)\b(FDIC|OCC|Federal Reserve|FASB|U\.S\.|United States|SEC)\b", full_search):
        jurisdiction = "US"
    elif re.search(r"(?i)\b(SAMA|Saudi|Riyadh|KSA|SAR)\b", full_search):
        jurisdiction = "Saudi Arabia"
    elif re.search(r"(?i)\b(BCBS|Basel|BIS)\b", full_search):
        jurisdiction = "Basel / Global"
    elif re.search(r"(?i)\b(Aldar|UAE|Abu Dhabi|AED)\b", full_search):
        jurisdiction = "UAE"

    doc_type = "other"
    if re.search(r"(?i)\bAnnual Report\b", full_search):
        doc_type = "annual report"
    elif re.search(r"(?i)\b(Guidance|Guidelines)\b", full_search):
        doc_type = "supervisory guidance"
    elif re.search(r"(?i)\bHandbook\b", full_search):
        doc_type = "handbook"
    elif re.search(r"(?i)\b(Update|Standard|Topic 326|ASU)\b", full_search):
        doc_type = "accounting standard"
    elif re.search(r"(?i)\b(Memo|Memorandum)\b", full_search):
        doc_type = "memo"
    elif re.search(r"(?i)\bFact Sheet\b", full_search):
        doc_type = "fact sheet"
    elif re.search(r"(?i)\b(Federal Register|Proposal|Proposed Rule|91 FR)\b", full_search):
        doc_type = "regulation/proposal"
    elif re.search(r"(?i)\b(Research|Working Paper|doi\.org|arXiv)\b", full_search):
        doc_type = "research paper"

    return {
        "title": clean_base,
        "entity": entity,
        "doc_type": doc_type,
        "period": ", ".join(yrs),
        "jurisdiction": jurisdiction,
        "topics": [],
        "summary": head[:400].replace("\n", " ")
    }

def build_card(fn, pages, llm):
    card = heuristic_card(fn, pages)
    if llm is not None:
        try:
            head = "\n".join(p[:1800] for p in pages[:3] if p.strip())[:5000]
            r = llm.json(CARD_SYS, CARD_USER.format(fn=fn, n=len(pages), head=head), max_tokens=800, name="doc_card")["json"]
            card.update({k: v for k, v in r.items() if v})
        except Exception: pass
    card["filename"] = fn; card["n_pages"] = len(pages)
    return card

# ---------------- index build ----------------
def coll_dir(name): return os.path.join(C.DATA_DIR, re.sub(r"[^\w\-.]", "_", name))

def index_zip(zip_bytes_path, collection, llm=None, progress=print):
    d = coll_dir(collection)
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    raw = os.path.join(d, "raw"); os.makedirs(raw)
    with zipfile.ZipFile(zip_bytes_path) as z: z.extractall(raw)
    pdfs = sorted(glob.glob(os.path.join(raw, "**", "*.pdf"), recursive=True), key=lambda p: os.path.basename(p))
    docs, chunks = [], []
    def work(p):
        fn = os.path.basename(p); pages = extract_pages(p)
        return fn, pages
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(work, pdfs))
    with ThreadPoolExecutor(max_workers=8) as ex:      # cards = 1 cheap LLM call per document
        cards = list(ex.map(lambda r: build_card(r[0], r[1], llm), results))
    for did, ((fn, pages), card) in enumerate(zip(results, cards)):
        card["doc_id"] = did; docs.append(card)
        for pno, t in enumerate(pages, 1):
            for c in chunk_page(t, fn, pno, C.CHUNK_CHARS): c["doc_id"] = did; chunks.append(c)
        progress(f"indexed {fn}: {len(pages)} pages")
    for i, c in enumerate(chunks): c["id"] = i
    shutil.rmtree(raw, ignore_errors=True)
    from .retrieve import build_matrices
    store = {"docs": docs, "chunks": chunks}
    store.update(build_matrices(docs, chunks))
    with open(os.path.join(d, "store.pkl"), "wb") as f: pickle.dump(store, f)
    return {"documents": len(docs), "chunks": len(chunks)}
