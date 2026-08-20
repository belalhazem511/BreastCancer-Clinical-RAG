from pathlib import Path
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi

# 1. File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
METADATA_FILE = DATA_DIR / "chunks_metadata.json"
VECTORS_FILE = DATA_DIR / "chunk_vectors.npy"

# Standard guideline titles
GUIDELINE_TITLES = {
    "NG101": "NICE Guideline NG101 — Early and locally advanced breast cancer: diagnosis and management",
    "CG81": "NICE Guideline CG81 — Advanced breast cancer: diagnosis and treatment",
    "CG164": "NICE Guideline CG164 — Familial breast cancer: classification, care and managing breast cancer and related risks in people with a family history of breast cancer",
}

# 2. Load chunks + metadata
with open(METADATA_FILE, "r", encoding="utf-8") as file:
    documents = json.load(file)

chunks = []
for document in documents:
    metadata = document["metadata"]
    source_path = metadata["source"]
    
    # Determine canonical source name
    if "NG101" in source_path:
        canonical_name = GUIDELINE_TITLES["NG101"]
    elif "CG81" in source_path:
        canonical_name = GUIDELINE_TITLES["CG81"]
    elif "CG164" in source_path:
        canonical_name = GUIDELINE_TITLES["CG164"]
    else:
        canonical_name = metadata.get("source_name", source_path)

    chunks.append({
        "chunk_id": metadata["chunk_id"],
        "source": metadata["source"],
        "source_name": canonical_name,
        "header": metadata["header"],
        "number": metadata["number"],
        "subheader": metadata["subheader"],
        "start_page": metadata["start_page"],
        "end_page": metadata["end_page"],
        "chunk_number": metadata["chunk_number"],
        "text": document["text"]
    })

# 3. Load saved BGE chunk vectors
embeddings = np.load(VECTORS_FILE)

# Make sure number of vectors matches number of chunks
if len(embeddings) != len(chunks):
    raise ValueError(
        "Number of embeddings does not match number of chunks. "
        "Run Embeddings.py again."
    )

# 4. Load BGE model with ultra-low memory FastEmbed ONNX engine (<50MB RAM)
print("[*] Initializing ultra-low memory BGE embedding model (BAAI/bge-small-en-v1.5)...", flush=True)
_fastembed_model = None
_sentence_transformer_model = None

try:
    from fastembed import TextEmbedding
    _fastembed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    print("[+] FastEmbed ONNX runtime engine loaded successfully (RAM: <50MB).", flush=True)
except Exception as e:
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("[+] SentenceTransformer engine loaded successfully.", flush=True)
    except Exception as e2:
        print(f"[!] Warning: Embedding model could not be loaded: {e2}", flush=True)


def encode_query_vector(text: str) -> np.ndarray:
    """Encode a single query string into a normalized 384-d vector with minimal RAM."""
    if _fastembed_model is not None:
        vecs = list(_fastembed_model.embed([text]))
        vec = np.asarray(vecs[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    elif _sentence_transformer_model is not None:
        try:
            vec = _sentence_transformer_model.encode_query(text, normalize_embeddings=True)
        except AttributeError:
            vec = _sentence_transformer_model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
    else:
        return np.zeros((384,), dtype=np.float32)

# 5. Prepare text for BM25
texts_for_bm25 = []
for chunk in chunks:
    header = chunk["header"] or ""
    subheader = chunk["subheader"] or ""
    chunk_text = chunk["text"]
    text = f"{header}\n{subheader}\n{chunk_text}"
    texts_for_bm25.append(text)

# 6. Stop words and comprehensive medical synonym mappings
stop_words = {
    "the", "is", "a", "an", "what", "of", "to", "for", "with", "how",
    "should", "be", "in", "on", "and", "or", "are", "was", "were", "do",
    "does", "did", "can", "could", "would", "at", "by", "from", "about",
    "when", "why", "which", "who", "whom", "will", "shall"
}

SYNONYMS = {
    "dcis": "ductal carcinoma in situ",
    "er+": "er positive estrogen receptor positive",
    "er-": "er negative estrogen receptor negative",
    "her2+": "her2 positive human epidermal growth factor receptor 2",
    "her2-": "her2 negative",
    "pr+": "progesterone receptor positive",
    "pr-": "progesterone receptor negative",
    "brca": "brca1 brca2 mutation genetic carrier",
    "brca1": "brca1 genetic carrier mutation",
    "brca2": "brca2 genetic carrier mutation",
    "tp53": "tp53 li-fraumeni syndrome mutation carrier",
    "chemo": "chemotherapy systemic therapy",
    "rt": "radiotherapy radiation",
    "radiation": "radiotherapy",
    "lumpectomy": "breast-conserving surgery wide local excision",
    "bcs": "breast-conserving surgery wide local excision",
    "mastectomy": "mastectomy surgical resection",
    "slnb": "sentinel lymph node biopsy axilla axillary",
    "anc": "axillary node clearance axillary lymph node dissection",
    "ai": "aromatase inhibitor anastrozole letrozole exemestane",
    "ais": "aromatase inhibitors anastrozole letrozole exemestane",
    "tam": "tamoxifen selective estrogen receptor modulator",
    "ofs": "ovarian function suppression gnrh lhrh agonist goserelin",
    "menopause": "postmenopausal premenopausal menopausal status",
    "premenopausal": "premenopausal women younger ovarian function",
    "postmenopausal": "postmenopausal women aromatase inhibitor",
    "bone": "bisphosphonates bone health zoledronic acid sodium clodronate bone metastases",
    "bisphosphonates": "bisphosphonates zoledronic acid sodium clodronate bone health",
    "staging": "staging distant metastases imaging investigations pretreatment assessment cect ct mri pet-ct bone scan",
    "scan": "imaging assessment cect ct mri ultrasound mammography pet-ct bone scintigraphy",
    "scans": "imaging assessment cect ct mri ultrasound mammography pet-ct bone scintigraphy",
    "mri": "magnetic resonance imaging mri breast surveillance preoperative assessment",
    "pet": "fdg pet-ct positron emission tomography imaging",
    "pet-ct": "fdg pet-ct positron emission tomography imaging",
    "surveillance": "surveillance monitoring follow-up imaging mammography annual",
    "follow-up": "follow-up surveillance annual mammography care plan",
    "followup": "follow-up surveillance annual mammography care plan",
    "margins": "surgical resection margins 2mm radial margin ink on tumor 0mm",
    "margin": "surgical resection margin 2mm radial margin ink on tumor 0mm",
    "chemoprevention": "chemoprevention risk reduction tamoxifen anastrozole raloxifene familial risk",
    "prevention": "chemoprevention risk reduction tamoxifen anastrozole raloxifene risk-reducing surgery",
    "prophylactic": "risk-reducing mastectomy risk-reducing salpingo-oophorectomy bilateral mastectomy",
    "male": "male breast cancer men ER-positive reproductive organs",
    "men": "male breast cancer men ER-positive reproductive organs tamoxifen",
    "genetic": "genetic testing brca1 brca2 tp53 carrier probability threshold ten percent",
    "hereditary": "familial breast cancer family history moderate risk high risk carrier",
    "familial": "familial breast cancer family history moderate risk high risk lifetime risk",
    "asymptomatic": "asymptomatic early breast cancer routine staging investigations",
    "negative": "do not offer do not routinely use not recommended against contraindication",
    "trastuzumab": "trastuzumab herceptin her2 targeted therapy pertuzumab t-dm1",
    "reconstruction": "breast reconstruction immediate delayed implant autologous latissimus dorsi diep flap",
    "fertility": "fertility preservation pregnancy oocyte cryopreservation ovarian function suppression",
    "lymphoedema": "lymphoedema upper limb swelling physiotherapy compression",
    "lymphedema": "lymphoedema upper limb swelling physiotherapy compression",
    "side-effects": "side effects menopausal symptoms hot flushes arthralgia osteoporosis",
    "side effects": "side effects menopausal symptoms hot flushes arthralgia osteoporosis",
}

# Domain vocabulary specifically for NICE breast cancer guidelines (NG101, CG81, CG164)
BREAST_DOMAIN_KEYWORDS = {
    "breast", "mammary", "mastectomy", "lumpectomy", "mammogram", "mammography",
    "dcis", "lcis", "ductal", "lobular", "tamoxifen", "anastrozole", "letrozole",
    "exemestane", "trastuzumab", "herceptin", "pertuzumab", "perjeta", "t-dm1",
    "kadcyla", "palb2", "brca", "brca1", "brca2", "tp53", "slnb", "sentinel lymph node",
    "axilla", "axillary", "gynaecomastia", "gynecomastia", "paget", "filgrastim",
    "zoledronic", "clodronate", "re-excision", "breast-conserving", "bcs",
    "ng101", "cg81", "cg164", "familial breast", "family history of breast"
}

# Explicit non-breast cancer & out-of-scope clinical patterns
OUT_OF_SCOPE_PATTERNS = [
    # Non-breast cancers & organs
    r"\b(?:lung cancer|nsclc|sclc|non-small cell|small cell lung)\b",
    r"\b(?:prostate cancer|prostate carcinoma|psa screening|prostatectomy)\b",
    r"\b(?:colon cancer|colorectal|bowel cancer|rectal cancer|folfox|folfiri)\b",
    r"\b(?:pancreatic cancer|pancreas adenocarcinoma|whipple)\b",
    r"\b(?:glioblastoma|glioma|brain tumor|brain tumour|astrocytoma)\b",
    r"\b(?:melanoma|skin cancer|basal cell|squamous cell carcinoma)\b",
    r"\b(?:cervical cancer|cervix carcinoma|pap smear|hpv vaccine)\b",
    r"\b(?:endometrial cancer|endometrium carcinoma|uterine cancer|uterus cancer)\b",
    r"\b(?:renal cell|kidney cancer|nephrectomy)\b",
    r"\b(?:bladder cancer|urothelial)\b",
    r"\b(?:leukemia|leukaemia|acute myeloid|aml|all|cll|cml)\b",
    r"\b(?:lymphoma|hodgkin|non-hodgkin|diffuse large b-cell)\b",
    r"\b(?:multiple myeloma|myeloma)\b",
    r"\b(?:gastric cancer|stomach cancer|gastrectomy)\b",
    r"\b(?:esophageal cancer|oesophageal)\b",
    r"\b(?:liver cancer|hepatocellular|hcc|cholangiocarcinoma)\b",
    r"\b(?:thyroid cancer|papillary thyroid)\b",
    r"\b(?:sarcoma|osteosarcoma|chondrosarcoma|ewing)\b",
    r"\b(?:epithelial ovarian cancer|ovarian cancer treatment|debulking surgery)\b",
    # Non-oncology diseases
    r"\b(?:diabetes|diabetic|metformin|insulin glargine|sglt2|hba1c)\b",
    r"\b(?:acute coronary syndrome|myocardial infarction|heart attack|angina|stent)\b",
    r"\b(?:hypertension|blood pressure|lisinopril|amlodipine)\b",
    r"\b(?:stroke|ischemic stroke|tpa|alteplase)\b",
    r"\b(?:asthma|copd|inhaler|albuterol)\b",
    r"\b(?:alzheimer|dementia|parkinson)\b",
    r"\b(?:covid-19|sars-cov-2|coronavirus)\b",
    # Foreign / non-NICE guidelines
    r"\b(?:nccn|nccn guidelines|asco|asco guidelines|esmo|esmo guidelines|american cancer society)\b",
    # Pseudoscience / unproven alternative therapies
    r"\b(?:baking soda|alkaline diet|alkaline water|ozone therapy|high-dose vitamin c|intravenous vitamin c|vitamin c cure|ivermectin for cancer|apricot seeds|amygdalin|laetrile|coffee enema)\b",
]

def is_hard_out_of_scope(question: str) -> bool:
    """Check if the query is explicitly asking about non-breast cancer, foreign guidelines, or non-medical topics."""
    q_lower = question.lower()
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q_lower):
            # Special exception: CG164 covers familial risk in families with history of breast and ovarian cancer
            if "ovarian" in pat and any(k in q_lower for k in ["familial", "family history", "brca", "genetic", "risk"]):
                if not any(k in q_lower for k in ["treatment", "chemotherapy", "cisplatin", "carboplatin", "debulking", "surgery"]):
                    continue
            return True
    return False

# 7. BM25 tokenizer with synonym expansion
def tokenize(text, expand_synonyms=False):
    text_lower = text.lower()
    
    # Expand medical abbreviations if requested (for queries)
    if expand_synonyms:
        for abbr, expansion in SYNONYMS.items():
            text_lower = re.sub(rf"\b{re.escape(abbr)}\b", expansion, text_lower)
            
    # Extract alphanumeric words, numbers, percentages, and section identifiers
    words = re.findall(r"\d+(?:\.\d+)*%?|[a-z0-9]+(?:-[a-z0-9]+)*", text_lower)
    
    # Remove common stop words
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return words

# 8. Tokenize all chunks
tokenized_chunks = [tokenize(text, expand_synonyms=False) for text in texts_for_bm25]

# 9. Build BM25 index
bm25 = BM25Okapi(tokenized_chunks)

# 10. Normalization helpers
def normalize_scores(scores):
    """Normalize raw scores into [0, 1] range safely."""
    minimum = np.min(scores)
    maximum = np.max(scores)
    if maximum == minimum:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - minimum) / (maximum - minimum)

# 11. Hybrid retrieval with Reciprocal Rank Fusion & Dynamic Confidence
def hybrid_query(
    question,
    top_k=5,
    semantic_weight=0.65,
    keyword_weight=0.35,
    source_filter=None
):
    """
    Perform calibrated hybrid retrieval combining BGE dense semantic search
    and BM25 keyword matching with Reciprocal Rank Fusion (RRF).
    """
    # 1. HARD OUT-OF-SCOPE GATING
    if is_hard_out_of_scope(question):
        return []

    # A. SEMANTIC RETRIEVAL - BGE
    query_embedding = encode_query_vector(question)
    semantic_scores = embeddings @ query_embedding

    # B. KEYWORD RETRIEVAL - BM25 with query expansion
    query_tokens = tokenize(question, expand_synonyms=True)
    if not query_tokens:
        query_tokens = tokenize(question, expand_synonyms=False)
        
    keyword_scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)

    best_semantic_score = float(np.max(semantic_scores))
    best_keyword_score = float(np.max(keyword_scores)) if len(keyword_scores) > 0 else 0.0

    # C. PRE-LLM RELEVANCE GATING
    q_lower = question.lower()
    has_breast_keyword = any(kw in q_lower for kw in BREAST_DOMAIN_KEYWORDS)

    if not has_breast_keyword:
        # Non-breast queries must meet strong semantic (>=0.62) and keyword evidence to pass
        if best_semantic_score < 0.62 and best_keyword_score < 3.5:
            return []
    else:
        # Breast queries are preserved for boundary evaluation unless catastrophically low (<0.25)
        if best_semantic_score < 0.25 and best_keyword_score < 0.5:
            return []

    # D. CALIBRATED SCORE NORMALIZATION
    # Semantic: Cosine similarity mapped linearly from [0.35, 0.90] to [0.0, 1.0]
    semantic_normalized = np.clip((semantic_scores - 0.35) / 0.55, 0.0, 1.0)
    
    # Keyword: BM25 score soft-saturated
    keyword_normalized = keyword_scores / (keyword_scores + 10.0)

    # E. RECIPROCAL RANK FUSION (RRF)
    sem_order = np.argsort(semantic_scores)[::-1]
    sem_ranks = np.empty_like(sem_order)
    sem_ranks[sem_order] = np.arange(len(semantic_scores))

    kw_order = np.argsort(keyword_scores)[::-1]
    kw_ranks = np.empty_like(kw_order)
    kw_ranks[kw_order] = np.arange(len(keyword_scores))

    k = 60
    rrf_scores = (semantic_weight / (k + sem_ranks + 1)) + (keyword_weight / (k + kw_ranks + 1))
    
    # Combined hybrid confidence score in [0, 1]
    hybrid_scores = (semantic_weight * semantic_normalized) + (keyword_weight * keyword_normalized)

    # F. RANK CANDIDATES
    sorted_indices = np.argsort(rrf_scores)[::-1]

    # Optional source filter
    if source_filter and source_filter.strip():
        sf = source_filter.strip().lower()
        sorted_indices = [
            idx for idx in sorted_indices
            if sf in chunks[idx]["source"].lower() or sf in chunks[idx]["source_name"].lower()
        ]

    top_indices = sorted_indices[:top_k]

    # G. BUILD RESULTS
    results = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]
        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "hybrid_score": float(round(float(hybrid_scores[index]), 3)),
            "rrf_score": float(round(float(rrf_scores[index]), 4)),
            "semantic_score": float(round(float(semantic_scores[index]), 3)),
            "semantic_normalized": float(round(float(semantic_normalized[index]), 3)),
            "keyword_score": float(round(float(keyword_scores[index]), 2)),
            "keyword_normalized": float(round(float(keyword_normalized[index]), 3)),
            "source": chunk["source"],
            "source_name": chunk["source_name"],
            "header": chunk["header"],
            "section": chunk["number"],
            "section_name": chunk["subheader"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"]
        })
    return results


def get_guideline_sources():
    """Return summary metadata for all connected clinical guidelines."""
    source_stats = {}
    for chunk in chunks:
        src = chunk["source"]
        if src not in source_stats:
            short = "NG101" if "NG101" in src else ("CG81" if "CG81" in src else ("CG164" if "CG164" in src else Path(src).stem))
            source_stats[src] = {
                "source": src,
                "source_name": chunk["source_name"],
                "short_name": short,
                "filename": Path(src).name,
                "chunk_count": 0,
                "sections": set(),
                "min_page": chunk["start_page"],
                "max_page": chunk["end_page"],
            }
        source_stats[src]["chunk_count"] += 1
        if chunk["number"]:
            source_stats[src]["sections"].add(str(chunk["number"]))
        source_stats[src]["min_page"] = min(source_stats[src]["min_page"], chunk["start_page"])
        source_stats[src]["max_page"] = max(source_stats[src]["max_page"], chunk["end_page"])

    result = []
    for src, data in source_stats.items():
        result.append({
            "id": data["short_name"],
            "shortSource": data["short_name"],
            "source": f"NICE {data['short_name']}",
            "source_name": data["source_name"],
            "filename": data["filename"],
            "pdf_url": f"/api/pdf/{data['filename']}",
            "chunk_count": data["chunk_count"],
            "section_count": len(data["sections"]),
            "page_range": f"{data['min_page']}–{data['max_page']}",
            "status": "Connected"
        })
    return result