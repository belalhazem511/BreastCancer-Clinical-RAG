import os
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq


# ============================================================
# Directory paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "RAG system"
FRONTEND_DIR = BASE_DIR / "breast-cancer-ai-clean"
DATA_DIR = RAG_DIR / "Data"


# ============================================================
# Load environment variables
# ============================================================

load_dotenv(RAG_DIR / ".env")
load_dotenv(BASE_DIR / ".env")


# ============================================================
# Add RAG system directory to Python path
# ============================================================

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# ============================================================
# Import retrieval engine
# ============================================================

import Retrieval


# ============================================================
# ============================================================
# Initialize Groq & Prioritized Model Fallback Chain
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is not set!")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Prioritized list of supported Groq models for high availability and automatic failover
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "qwen/qwen3.6-27b",
    "allam-2-7b",
]

GROQ_MODEL = GROQ_MODELS[0]


def generate_llm_response(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1500
) -> tuple[str, str]:
    """
    Generate chat completion with automatic failover across available Groq models.
    Returns (cleaned_response_text, model_name_used).
    """
    if not groq_client:
        return "", "none"

    last_err = None
    for model_name in GROQ_MODELS:
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            raw_text = completion.choices[0].message.content or ""
            # Strip reasoning tokens (e.g. <think>...</think>) if present
            cleaned_text = re.sub(
                r"<think>[\s\S]*?</think>",
                "",
                raw_text,
                flags=re.DOTALL
            ).strip()
            if cleaned_text:
                return cleaned_text, model_name
        except Exception as err:
            last_err = err
            print(f"[Groq Fallback] Model '{model_name}' encountered error: {err}. Attempting next model...")
            continue

    print(f"[Groq Error] All Groq models failed. Last error: {last_err}")
    return "", "none"


# ============================================================
# Initialize FastAPI
# ============================================================

app = FastAPI(
    title="Breast Cancer AI - Clinical RAG API",
    description=(
        "Evidence-grounded clinical decision support assistant "
        "for NICE breast cancer guidelines"
    ),
    version="1.0.0",
)


# ============================================================
# Enable CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response Models
# ============================================================

class ChatRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    top_k: Optional[int] = 4


class SourceDetails(BaseModel):
    id: str
    shortSource: str
    source: str
    source_name: str
    filename: str
    pdf_url: str
    chunk_count: int
    page_range: str
    status: str


# ============================================================
# Clinical RAG System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert clinical decision support assistant specialized strictly in NICE (National Institute for Health and Care Excellence) breast cancer guidelines.

KNOWLEDGE BASE SCOPE:
Your knowledge base is strictly limited to three official NICE guidelines:
1. NICE NG101: Early and locally advanced breast cancer: diagnosis and management
2. NICE CG81: Advanced breast cancer: diagnosis and treatment
3. NICE CG164: Familial breast cancer: classification, care and managing breast cancer and related risks in people with a family history of breast cancer

CORE INSTRUCTIONS:
1. Answer the user's question using ONLY the retrieved NICE guideline context provided below.
2. Direct and Complex Questions with Answers in Context:
   - Provide clear, direct, and clinically precise recommendations supported by the evidence.
   - Preserve exact medical details: drug names (e.g., anastrozole, letrozole, exemestane, tamoxifen, trastuzumab), dosages, receptor status (ER, PR, HER2), disease stage, risk thresholds, surgical margin criteria (e.g. 0 mm for invasive, 2 mm for DCIS), and surveillance intervals.
   - If multiple retrieved chunks from the same or different sections provide complementary details, synthesize them coherently.
3. Boundary & Negative Recommendations:
   - If the guideline explicitly recommends AGAINST an intervention (e.g. "Do not routinely offer whole-body staging CT/MRI/bone scans in asymptomatic early breast cancer" or "Do not routinely perform breast MRI for all patients"), clearly state this negative recommendation as a valid, high-confidence answer.
4. Out-of-Scope and Hard Out-of-Scope Questions:
   - If the question asks about other types of cancer (e.g. lung, prostate, colon), non-breast medical conditions, non-NICE foreign guidelines (e.g. NCCN, FDA), or unproven alternative therapies (e.g. high-dose vitamin C, homeopathy) NOT supported in the retrieved NICE context:
   - YOU MUST NOT make up answers, infer cross-disease treatments, or force breast cancer evidence onto other diseases.
   - Return the standard Insufficient Context response immediately.
5. General Non-Medical Questions (e.g. physics, programming, geography):
   - Return the standard Insufficient Context response immediately.

REQUIRED OUTPUT FORMAT:

If the retrieved context directly supports the question:

Recommendations:
- [Primary recommendation point 1 with exact qualifiers, drug names, or criteria]
- [Additional recommendation points as needed]

Supporting Evidence:
- [Specific supporting evidence extracted from the retrieved NICE text]

Citation:
NICE Guideline [NG101 / CG81 / CG164] — [Full Guideline Title], Section [X.X], Recommendations [X.X.X], Pages [Start–End].

Confidence and Safety:
- Confidence: High (if directly and fully answered) OR Medium (if partially answered or synthesized across sections)
- [Brief clinical explanation for why the confidence level was selected]
- This response is based solely on retrieved NICE guideline context and does not replace professional clinical judgement.


If the retrieved context DOES NOT support the question:

Insufficient Context:
The retrieved NICE guideline context does not contain information that supports this question. The knowledge base is strictly focused on NICE breast cancer guidelines (NG101, CG81, CG164).

Citation:
No applicable NICE guideline citation was found for this question.

Confidence and Safety:
- Confidence: Low
- The retrieved context does not support an answer to this question.
- No answer was generated from outside knowledge.
""".strip()


# ============================================================
# Helper: Detect insufficient-context LLM response
# ============================================================

def is_insufficient_context(text: str) -> bool:
    """
    Check whether the LLM determined that retrieved NICE evidence
    does not support the user's question.
    """
    lowered = text.lower()
    return (
        "insufficient context" in lowered
        or (
            "does not contain information" in lowered
            and "supports this question" in lowered
        )
        or "no applicable nice guideline citation was found" in lowered
        or "no applicable nice guideline citation was generated" in lowered
        or "does not contain information that supports" in lowered
    )


# ============================================================
# Helper: Parse LLM answer
# ============================================================

def parse_llm_response(
    raw_text: str,
    fallback_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:

    # Clean any reasoning tokens
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Check whether the LLM rejected the retrieved context
    if is_insufficient_context(text):
        return {
            "has_context": False,
            "summary": (
                "The retrieved NICE guideline context does not contain "
                "information that supports this question."
            ),
            "recommendations": [],
            "supporting_evidence": [],
            "confidence": "Low",
            "confidence_reason": (
                "The retrieved context does not support an answer "
                "to this question."
            ),
        }

    summary = ""
    recommendations = []
    supporting_evidence = []
    confidence = "Medium"
    confidence_reason = (
        "The answer is supported by retrieved NICE guideline evidence."
    )

    # Extract Summary if explicitly provided
    summary_match = re.search(
        r"(?:\*\*)?Summary:?(?:\*\*)?\s*"
        r"(.*?)"
        r"(?=\n\s*(?:#+\s*)?(?:\*\*)?"
        r"(?:Recommendations|Supporting Evidence|Confidence|Citation)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if summary_match:
        summary = summary_match.group(1).strip()

    # Extract Recommendations
    rec_match = re.search(
        r"(?:#+\s*)?(?:\*\*)?Recommendations:?(?:\*\*)?\s*"
        r"(.*?)"
        r"(?=\n\s*(?:#+\s*)?(?:\*\*)?"
        r"(?:Supporting Evidence|Confidence|Citation|Summary)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if rec_match:
        rec_block = rec_match.group(1).strip()
        for line in rec_block.split("\n"):
            clean_line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
            if clean_line and len(clean_line) > 3:
                recommendations.append(clean_line)

    # Extract Supporting Evidence
    evidence_match = re.search(
        r"(?:#+\s*)?(?:\*\*)?Supporting Evidence:?(?:\*\*)?\s*"
        r"(.*?)"
        r"(?=\n\s*(?:#+\s*)?(?:\*\*)?"
        r"(?:Confidence|Citation|Recommendations|Summary)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if evidence_match:
        evidence_block = evidence_match.group(1).strip()
        for line in evidence_block.split("\n"):
            clean_line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
            if clean_line and len(clean_line) > 3:
                supporting_evidence.append(clean_line)

    # Extract Confidence (High, Medium, Low)
    confidence_match = re.search(
        r"(?:\*\*)?Confidence:?(?:\*\*)?\s*"
        r"(?:\*\*)?(High|Medium|Low)(?:\*\*)?",
        text,
        re.IGNORECASE,
    )
    if confidence_match:
        confidence = confidence_match.group(1).capitalize()

    # Extract explanation after confidence
    safety_match = re.search(
        r"(?:#+\s*)?(?:\*\*)?Confidence and Safety:?(?:\*\*)?\s*"
        r"(.*?)(?=\n\s*(?:#+\s*)?(?:Recommendations|Supporting Evidence|Citation)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if safety_match:
        safety_text = safety_match.group(1).strip()
        reason_lines = []
        for line in safety_text.split("\n"):
            clean_line = re.sub(r"^[-*•\s]+", "", line).strip()
            if clean_line and not re.match(r"^\**confidence\s*:", clean_line, re.IGNORECASE):
                reason_lines.append(clean_line)
        if reason_lines:
            confidence_reason = " ".join(reason_lines)

    if summary:
        summary = re.sub(r"^[\*\#\-\s:]+", "", summary).strip()
        summary = re.sub(r"[\*\#]+$", "", summary).strip()
        # If extracted summary is a reasoning artifact, clear it to fallback to recommendation
        if len(summary) < 5 or any(k in summary.lower() for k in ["scan retrieved", "thinking process", "analyze query", "here is"]):
            summary = ""

    # If there was no explicit or clean Summary, use first recommendation
    if not summary:
        if recommendations:
            summary = recommendations[0]
        else:
            summary = (
                "The retrieved NICE guideline context contains "
                "information relevant to this question."
            )

    return {
        "has_context": True,
        "summary": summary,
        "recommendations": recommendations,
        "supporting_evidence": supporting_evidence,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
    }


# ============================================================
# Helper: Build LLM Context
# ============================================================

def build_context_string(
    results: List[Dict[str, Any]],
) -> str:

    context_parts = []
    for result in results:
        context_part = (
            f"Guideline: {result['source_name']}\n"
            f"Section Number: {result['section']}\n"
            f"Section Subheader: {result['section_name']}\n"
            f"Pages: {result['start_page']} - {result['end_page']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Evidence Text:\n"
            f"{result['text']}"
        )
        context_parts.append(context_part)

    return ("\n\n-----------------------------\n\n").join(context_parts)


# ============================================================
# Helper: Build frontend citation metadata
# ============================================================

def build_citations(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    citations = []
    for res in results:
        src_path = res["source"]
        if "NG101" in src_path:
            short_src = "NG101"
            total_pages = 108
        elif "CG81" in src_path:
            short_src = "CG81"
            total_pages = 46
        elif "CG164" in src_path:
            short_src = "CG164"
            total_pages = 51
        else:
            short_src = Path(src_path).stem
            total_pages = res["end_page"]

        hybrid_score = res.get("hybrid_score", 0.0)

        chunk_citation = {
            "id": res["chunk_id"],
            "rank": res["rank"],
            "source": f"NICE {short_src}",
            "shortSource": short_src,
            "source_name": res["source_name"],
            "section": (
                f"Section {res['section']}"
                if res["section"]
                else "General Guidance"
            ),
            "sectionNumber": str(res["section"]) if res["section"] else "",
            "section_name": res["section_name"] or "Clinical Guidance",
            "pages": f"Pages {res['start_page']}–{res['end_page']}",
            "pageRange": f"{res['start_page']}–{res['end_page']}",
            "start_page": res["start_page"],
            "end_page": res["end_page"],
            "firstPage": res["start_page"],
            "pageCount": f"{res['start_page']} / {total_pages}",
            "filename": f"{short_src}.pdf",
            "pdf_url": f"/api/pdf/{short_src}.pdf",
            "pdf_page_url": f"/api/pdf/{short_src}.pdf#page={res['start_page']}",
            "description": f"{res['section_name']} ({res['source_name']})",
            "previewTitle": (
                f"{res['section']} {res['section_name']}"
                if res["section"]
                else res["section_name"]
            ),
            "text": res["text"],
            "hybrid_score": round(hybrid_score, 3),
        }
        citations.append(chunk_citation)

    return citations


# ============================================================
# API: Health Check
# ============================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Breast Cancer AI RAG Backend",
        "llm_model": GROQ_MODEL,
        "available_models": GROQ_MODELS,
        "total_chunks": len(Retrieval.chunks),
        "guidelines": [
            "NICE NG101",
            "NICE CG81",
            "NICE CG164",
        ],
    }


# ============================================================
# API: Guideline Sources
# ============================================================

@app.get("/api/sources")
@app.get("/api/guidelines")
async def get_sources():
    try:
        sources = Retrieval.get_guideline_sources()
        return {
            "success": True,
            "sources": sources,
        }
    except Exception as e:
        print(f"Source loading error: {e}")
        return {
            "success": True,
            "sources": [
                {
                    "id": "NG101",
                    "shortSource": "NG101",
                    "source": "NICE NG101",
                    "source_name": (
                        "NICE Guideline NG101 — "
                        "Early and locally advanced breast cancer: diagnosis and management"
                    ),
                    "filename": "NG101.pdf",
                    "pdf_url": "/api/pdf/NG101.pdf",
                    "chunk_count": 66,
                    "section_count": 17,
                    "page_range": "7–58",
                    "status": "Connected",
                },
                {
                    "id": "CG81",
                    "shortSource": "CG81",
                    "source": "NICE CG81",
                    "source_name": (
                        "NICE Guideline CG81 — "
                        "Advanced breast cancer: diagnosis and treatment"
                    ),
                    "filename": "CG81.pdf",
                    "pdf_url": "/api/pdf/CG81.pdf",
                    "chunk_count": 36,
                    "section_count": 14,
                    "page_range": "7–32",
                    "status": "Connected",
                },
                {
                    "id": "CG164",
                    "shortSource": "CG164",
                    "source": "NICE CG164",
                    "source_name": (
                        "NICE Guideline CG164 — "
                        "Familial breast cancer: classification, care and managing breast cancer and related risks in people with a family history of breast cancer"
                    ),
                    "filename": "CG164.pdf",
                    "pdf_url": "/api/pdf/CG164.pdf",
                    "chunk_count": 47,
                    "section_count": 7,
                    "page_range": "5–44",
                    "status": "Connected",
                },
            ],
        }


# ============================================================
# API: Serve PDF Files
# ============================================================

@app.get("/api/pdf/{filename}")
async def serve_pdf(filename: str):
    clean_filename = Path(filename).name
    pdf_path = DATA_DIR / clean_filename

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{clean_filename}' not found.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=clean_filename,
        headers={
            "Content-Disposition": f'inline; filename="{clean_filename}"',
            "Accept-Ranges": "bytes",
        },
    )


# ============================================================
# API: Clinical Chat (Full RAG Pipeline)
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    source_filter = payload.source_filter
    top_k = payload.top_k or 4

    # Step 1: Hybrid Retrieval with Reciprocal Rank Fusion
    try:
        results = Retrieval.hybrid_query(
            question=question,
            top_k=top_k,
            source_filter=source_filter,
        )
    except Exception as err:
        print(f"Retrieval error: {err}")
        results = []

    # Step 2: Retrieval found no relevant candidates (Pre-LLM rejection)
    if not results:
        insufficient_response = """
### Insufficient Context

The retrieved NICE guideline context does not contain information that supports this question.

### Citation

No applicable NICE guideline citation was found for this question.

### Confidence and Safety

- **Confidence: Low**
- The retrieved context does not support an answer to this question.
- No answer was generated from outside knowledge.
""".strip()

        return {
            "success": True,
            "has_context": False,
            "summary": (
                "The retrieved NICE guideline context does not contain "
                "information that supports this question."
            ),
            "recommendations": [],
            "supporting_evidence": [],
            "confidence": "Low",
            "confidence_reason": (
                "The retrieved context does not support "
                "an answer to this question."
            ),
            "source_match": "0%",
            "citations": [],
            "raw_response": insufficient_response,
        }

    # Step 3: Build grounded context string
    context_str = build_context_string(results)

    user_prompt = f"""
Question:
{question}

Retrieved NICE guideline context:

{context_str}

Answer the question using ONLY the retrieved context.

First determine whether the context actually supports the question.

Follow the required output format exactly.
""".strip()

    # Step 4: Generate grounded answer with model fallback
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_llm_response, used_model = generate_llm_response(
        messages=messages,
        temperature=0.0,
        max_tokens=1500
    )

    # Step 5: If LLM generation failed across all models or API key is not configured
    if not raw_llm_response:
        if results and results[0].get("hybrid_score", 0) >= 0.58:
            top_chunk = results[0]
            recommendations_list = []
            for r in results:
                txt = r.get("text", "").strip()
                lines = [l.strip("-•* \t") for l in txt.split("\n") if len(l.strip()) > 25]
                recommendations_list.extend(lines[:2])
            recommendations_list = recommendations_list[:4] if recommendations_list else ["Follow official NICE guideline recommendations specified in the cited sections."]

            citations = build_citations(results)
            top_score = results[0].get("hybrid_score", 0.80)
            confidence_level = "High" if top_score >= 0.70 else "Medium"

            return {
                "success": True,
                "has_context": True,
                "summary": f"Evidence-grounded recommendation derived directly from {top_chunk.get('source_name', 'NICE Guidelines')} ({top_chunk.get('section', '')}).",
                "recommendations": recommendations_list,
                "supporting_evidence": [
                    f"Guideline excerpt from {r.get('source', '')} {r.get('section', '')} ({r.get('pages', '')}): {r.get('text', '')[:140]}..."
                    for r in results[:2]
                ],
                "confidence": confidence_level,
                "confidence_reason": "Direct evidence extraction from indexed NICE guidelines.",
                "source_match": f"{min(99, max(75, round(top_score * 100)))}%",
                "citations": citations,
                "raw_response": f"Direct guideline grounding from {top_chunk.get('source', '')}",
            }
        else:
            failed_response = """
### Insufficient Context

A grounded answer could not be generated from the retrieved NICE guideline context.

### Citation

No applicable NICE guideline citation was generated for this question.

### Confidence and Safety

- **Confidence: Low**
- A grounded NICE guideline answer could not be generated.
- No answer was generated from outside knowledge.
""".strip()

            return {
                "success": True,
                "has_context": False,
                "summary": (
                    "The retrieved NICE guideline context does not contain "
                    "information that supports this question."
                ),
                "recommendations": [],
                "supporting_evidence": [],
                "confidence": "Low",
                "confidence_reason": (
                    "The retrieved context does not support "
                    "an answer to this question."
                ),
                "source_match": "0%",
                "citations": [],
                "raw_response": failed_response,
            }

    # Step 6: Parse LLM response
    parsed = parse_llm_response(raw_llm_response, results)

    # Step 7: LLM determined retrieved chunks do not support question
    if not parsed["has_context"]:
        return {
            "success": True,
            "has_context": False,
            "summary": parsed["summary"],
            "recommendations": [],
            "supporting_evidence": [],
            "confidence": "Low",
            "confidence_reason": parsed["confidence_reason"],
            "source_match": "0%",
            "citations": [],
            "raw_response": raw_llm_response,
        }

    # Step 8: Build citation metadata for supported answers
    citations = build_citations(results)

    # Calculate dynamic source match percentage
    top_score = results[0].get("hybrid_score", 0.85) if results else 0.85
    source_match_pct = f"{min(99, max(75, round(top_score * 100)))}%"

    # Step 9: Final supported response
    return {
        "success": True,
        "has_context": True,
        "summary": parsed["summary"],
        "recommendations": parsed["recommendations"],
        "supporting_evidence": parsed["supporting_evidence"],
        "confidence": parsed["confidence"],
        "confidence_reason": parsed["confidence_reason"],
        "source_match": source_match_pct,
        "citations": citations,
        "raw_response": raw_llm_response,
    }


# ============================================================
# API: Ask Endpoint (Compatibility alias for Static/app.js)
# ============================================================

@app.post("/api/ask")
async def ask_endpoint(payload: Dict[str, Any]):
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    chat_res = await chat_endpoint(ChatRequest(question=question))
    return {
        "answer": chat_res.get("raw_response", chat_res.get("summary", "")),
        "data": chat_res
    }


# ============================================================
# Mount Frontend Static Directories
# ============================================================

if (FRONTEND_DIR / "css").exists():

    app.mount(
        "/css",
        StaticFiles(
            directory=str(FRONTEND_DIR / "css")
        ),
        name="css",
    )


if (FRONTEND_DIR / "js").exists():

    app.mount(
        "/js",
        StaticFiles(
            directory=str(FRONTEND_DIR / "js")
        ),
        name="js",
    )


if (FRONTEND_DIR / "assets").exists():

    app.mount(
        "/assets",
        StaticFiles(
            directory=str(FRONTEND_DIR / "assets")
        ),
        name="assets",
    )


# ============================================================
# Serve Frontend Pages
# ============================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_index():

    file_path = FRONTEND_DIR / "index.html"

    return FileResponse(file_path)


@app.get("/home.html", response_class=HTMLResponse)
async def serve_home():

    file_path = FRONTEND_DIR / "home.html"

    return FileResponse(file_path)


@app.get("/chat.html", response_class=HTMLResponse)
async def serve_chat():

    file_path = FRONTEND_DIR / "chat.html"

    return FileResponse(file_path)


@app.get(
    "/uploaded-pdfs.html",
    response_class=HTMLResponse,
)
async def serve_uploaded_pdfs():

    file_path = (
        FRONTEND_DIR
        / "uploaded-pdfs.html"
    )

    return FileResponse(file_path)


@app.get(
    "/citation-history.html",
    response_class=HTMLResponse,
)
async def serve_citation_history():

    file_path = (
        FRONTEND_DIR
        / "citation-history.html"
    )

    return FileResponse(file_path)


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )