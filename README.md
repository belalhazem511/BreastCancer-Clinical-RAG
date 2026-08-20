---
title: BreastCancer.ai Clinical RAG
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Grounded NICE Guideline Clinical Decision Support with Groq LPU
---

# 🩺 BreastCancer.ai — Clinical RAG Decision-Support Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Groq Cloud](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en--v1.5-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![BM25](https://img.shields.io/badge/Search-BM25%20Okapi-0052CC?style=for-the-badge)](https://github.com/dorianbrown/rank_bm25)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An evidence-grounded clinical AI decision-support platform designed for breast oncology, strictly adhering to National Institute for Health and Care Excellence (NICE) guidelines.**

[System Architecture](#-system-architecture) • [Key Features](#-key-features) • [Quickstart Guide](#-quickstart-guide) • [API Reference](#-api-reference) • [Cloud Deployment](#-cloud-deployment--free-hosting) • [Verification Suite](#-automated-verification-suite)

</div>

---

## 📖 Overview

**BreastCancer.ai** is an end-to-end clinical AI decision-support system built to assist healthcare professionals, oncologists, and medical researchers. It bridges high-speed Large Language Model inference via **Groq** with strict, deterministic medical grounding from official **NICE Clinical Guidelines**:

- **NICE NG101**: Early and locally advanced breast cancer: diagnosis and management
- **NICE CG81**: Advanced breast cancer: diagnosis and treatment
- **NICE CG164**: Familial breast cancer: classification, care and managing risk in people with a family history of breast cancer

The system prevents hallucinations by employing a **hybrid dual-stage retrieval engine** (Dense Vector Embeddings + BM25 Okapi Lexical Search), dynamic similarity threshold scoring, structured schema generation, and source-level citation traceability down to the exact section, page number, and PDF document.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph UI ["Clinical Frontend Layer"]
        A[User / Clinician] -->|Enters Query| B["Clinical Workspace UI (Vanilla JS & Modern CSS)"]
        B -->|Async POST /api/chat| C[FastAPI REST API Gateway]
        B -->|View Supporting Evidence| D[Interactive Evidence Drawer]
        B -->|Open PDF Page| E["PDF Viewer (/api/pdf/{doc}#page=N)"]
    end

    subgraph Backend ["FastAPI Application Server"]
        C --> F{Query Processing}
        F --> G[Query Normalization & Medical Expansion]
    end

    subgraph Engine ["Hybrid RAG Engine (Retrieval.py)"]
        G --> H["Dense Vector Search (BAAI/bge-small-en-v1.5)"]
        G --> I["Lexical BM25 Search (Okapi BM25)"]
        H --> J["Min-Max Score Normalization"]
        I --> J
        J --> K["Hybrid Weighted Fusion (0.65 Dense + 0.35 Sparse)"]
        K --> L["Top-K Reranked Context Chunks & Metadata"]
    end

    subgraph LLM ["Clinical Grounding & Inference (Groq)"]
        L --> M{Similarity Threshold Check}
        M -->|Threshold Rejection (< 0.58)| N["Out-of-Context Safe Fallback"]
        M -->|Confidence Met (>= 0.58)| O["Medical Grounding Prompt Template"]
        O --> P["Groq Fast LPU Chain (GPT-OSS-120B / Qwen-27B / Compound)"]
        P --> Q["Structured JSON Response Parser"]
        Q --> R["Recommendations + Evidence + Confidence + Exact Citations"]
    end

    R --> C
    N --> C
```

---

## 🌟 Key Features

### 1. 🔍 Dual-Engine Hybrid Retrieval
- Combines semantic vector similarity using `BAAI/bge-small-en-v1.5` embeddings (cosine similarity) with exact keyword matching via **Rank-BM25 Okapi**.
- Applies min-max normalization and weighted score fusion (`0.65 * Dense + 0.35 * Sparse`) to capture both conceptual clinical context and exact medical terminology (drug names, receptor types, staging acronyms).

### 2. 🛡️ Anti-Hallucination Guardrails
- **Threshold Confidence Gate**: Queries without sufficient guideline context (score < 0.58) are safely rejected with transparent low-confidence disclaimers.
- **Strict Evidence Confinement**: The LLM prompt restricts answers solely to retrieved guideline chunks, strictly prohibiting unsupported medical assertions.

### 3. 📑 Exact Citation Traceability & Evidence Drawer
- Every clinical recommendation is accompanied by verifiable references: guideline name, chapter, exact section code (e.g., `NICE NG101 Section 1.11`), and target page numbers.
- Integrated **Evidence Drawer** allows clinicians to inspect the exact guideline chunk, toggle between chunks with `‹` and `›`, copy formatted citations to clipboard, and open the original PDF at the exact page.

### 4. ⚡ High-Speed Groq Failover Chain
- Backed by Groq LPU inference with automatic fallback across multi-model tiers (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `groq/compound-mini`, `qwen/qwen3.6-27b`, `allam-2-7b`) ensuring zero downtime and sub-second generation times.

### 5. 🏥 Complete Clinical Workspace
- **Clinical Suggestions**: Instant quick-start queries covering ER+/HER2+ therapies, genetic screening, lymph node staging, and surveillance.
- **Citation History**: Persistent traceability log of all evidence citations generated across sessions with search and export capabilities.
- **Guideline Library**: Live catalog with metadata, file size, page counts, and direct viewing for NICE NG101, CG81, and CG164.
- **Consultation Export**: One-click export of clinical consultations to Markdown (`.md`) with timestamps and citations.

---

## 📁 Repository Structure

```
.
├── server.py                   # FastAPI backend server & REST API router
├── run.py                      # One-click Python launcher with port detection & auto-browser
├── clean_port.py               # Port-conflict management and clean shutdown utility
├── run.bat                     # Windows batch launcher
├── verify_full_cycle.py        # Automated end-to-end verification test suite
├── requirements.txt            # Python production dependencies
├── Dockerfile                  # Multi-stage production container configuration (Hugging Face / Render / Koyeb)
├── docker-compose.yml          # Container orchestration configuration
├── render.yaml                 # 1-Click Render Cloud deployment blueprint
├── Procfile                    # PaaS process configuration
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git exclusion rules (safely excludes secrets)
│
├── .github/
│   └── workflows/
│       └── verify.yml          # Continuous Integration automated test pipeline
│
├── RAG system/                 # RAG core engine & knowledge base
│   ├── Retrieval.py            # Hybrid retrieval, normalization, and scoring
│   ├── Chunking_Metadata.py    # Guideline parser and page-aware chunking pipeline
│   ├── Parsing_Cleaning.py     # PDF text extraction and TOC structure parsing
│   ├── Embedding.py            # BGE vector generation script
│   ├── requirements.txt        # RAG-specific requirements
│   ├── .env.example            # Environment template for RAG module
│   └── Data/
│       ├── NG101.pdf           # NICE Early & Locally Advanced Breast Cancer (550 KB)
│       ├── CG81.pdf            # NICE Advanced Breast Cancer Guideline (198 KB)
│       ├── CG164.pdf           # NICE Familial Breast Cancer Guideline (255 KB)
│       ├── chunks_metadata.json# Pre-computed chunk metadata with section & page bounds
│       └── chunk_vectors.npy   # Pre-computed 384-dimensional BGE embedding vectors
│
└── breast-cancer-ai-clean/     # Frontend Web Interface
    ├── index.html              # Animated splash entry screen
    ├── home.html               # Clinical landing & search page
    ├── chat.html               # Interactive chat workspace with Evidence Drawer
    ├── uploaded-pdfs.html      # Guideline library & PDF preview modal
    ├── citation-history.html   # Evidence citation history log
    ├── assets/
    │   └── logo.svg            # Vector UI branding icon
    ├── css/
    │   ├── base.css            # Typography, variables & layout
    │   ├── splash.css          # Splash screen animations
    │   ├── home.css            # Landing page layout & cards
    │   ├── chat.css            # Chat interface & evidence drawer
    │   └── library.css         # Document grid & preview modal
    └── js/
        ├── splash.js           # Splash loader logic
        ├── home.js             # Query submission handler
        ├── chat.js             # Real-time RAG API integration & stream renderer
        ├── library.js          # Document viewer & PDF modal controller
        └── history.js          # Citation storage, search & clipboard copy
```

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.11+** installed
- A **Groq Cloud API Key** ([Get free key here](https://console.groq.com/keys))

### 1. Clone the Repository
```bash
git clone https://github.com/belalhazem511/BreastCancer-Clinical-RAG.git
cd BreastCancer-Clinical-RAG
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (or copy from `.env.example`):
```bash
cp .env.example .env
```
Edit `.env` and set your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
PORT=8000
```

### 4. Run the Application

#### Option A: One-Click Python Launcher (Recommended)
```bash
python run.py
```
*Automatically detects available ports, starts FastAPI, and opens your default browser at `http://127.0.0.1:8000`.*

#### Option B: Windows Batch Launcher
Double click `run.bat` or run:
```cmd
run.bat
```

#### Option C: Direct Uvicorn Command
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

#### Option D: Docker Container
```bash
docker-compose up --build
```

---

## 📡 API Reference

### `POST /api/chat`
Execute a clinical query grounded against NICE guidelines.

**Request Body:**
```json
{
  "question": "What is the recommended endocrine therapy for ER-positive early breast cancer?"
}
```

**Response Payload (`200 OK`):**
```json
{
  "success": true,
  "has_context": true,
  "confidence": "High",
  "source_match": "83%",
  "summary": "Offer adjuvant endocrine therapy to all patients with ER-positive invasive early breast cancer...",
  "recommendations": [
    "Offer tamoxifen as initial adjuvant endocrine therapy to premenopausal women.",
    "Offer an aromatase inhibitor (anastrozole or letrozole) to postmenopausal women at high risk of recurrence."
  ],
  "supporting_evidence": [
    "Adjuvant endocrine therapy significantly reduces 10-year breast cancer mortality in ER-positive disease."
  ],
  "citations": [
    {
      "source": "NICE NG101",
      "section": "Section 1.11",
      "pages": "Pages 32–33",
      "guideline_title": "Early and locally advanced breast cancer",
      "text": "1.11.1 Offer adjuvant endocrine therapy to people with ER-positive invasive early breast cancer...",
      "hybrid_score": 0.826,
      "pdf_url": "/api/pdf/NG101.pdf#page=32"
    }
  ]
}
```

---

### `GET /api/sources`
Retrieve metadata and connection status for all indexed clinical guidelines.

**Response (`200 OK`):**
```json
{
  "total_guidelines": 3,
  "sources": [
    {
      "code": "NG101",
      "title": "Early and locally advanced breast cancer: diagnosis and management",
      "type": "NICE Guideline",
      "status": "Connected & Indexed",
      "pages": 67,
      "size": "550 KB",
      "filename": "NG101.pdf"
    },
    {
      "code": "CG81",
      "title": "Advanced breast cancer: diagnosis and treatment",
      "type": "NICE Clinical Guideline",
      "status": "Connected & Indexed",
      "pages": 44,
      "size": "198 KB",
      "filename": "CG81.pdf"
    },
    {
      "code": "CG164",
      "title": "Familial breast cancer: classification, care and managing risk",
      "type": "NICE Clinical Guideline",
      "status": "Connected & Indexed",
      "pages": 52,
      "size": "255 KB",
      "filename": "CG164.pdf"
    }
  ]
}
```

---

### `GET /api/pdf/{filename}`
Streams the clinical guideline PDF file inline with browser viewing headers.

- **Example**: `GET /api/pdf/NG101.pdf`
- **Jump to page**: `/api/pdf/NG101.pdf#page=32`

---

### `GET /api/health`
Health check endpoint returning system status and Groq LLM connectivity.

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "engine": "Hybrid BGE + BM25",
  "groq_connected": true,
  "indexed_chunks": 426
}
```

---

## 🧪 Automated Verification Suite

The repository includes a comprehensive automated test suite testing all endpoints, UI static routes, PDF streaming, hybrid RAG scoring, and hallucination rejection:

```bash
python verify_full_cycle.py
```

**Test Coverage:**
- ✅ Static HTML Pages (`/`, `/home.html`, `/chat.html`, `/uploaded-pdfs.html`, `/citation-history.html`)
- ✅ CSS & JavaScript Asset Delivery
- ✅ REST API Health & Status Checks
- ✅ Clinical Guideline PDF Streaming (`NG101.pdf`, `CG81.pdf`, `CG164.pdf`)
- ✅ Grounded RAG Query Verification (NG101 Early Breast Cancer)
- ✅ Grounded RAG Query Verification (CG164 Familial / Genetic Risk)
- ✅ Out-of-Context Threshold Rejection & Hallucination Guardrail Check

---

## ☁️ Cloud Deployment & Free Hosting

### 1. Hugging Face Spaces (Free 100% Free CPU Docker)
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Space Name: `breastcancer-rag`
3. Space SDK: **Docker** (Blank)
4. Set Space visibility to **Public**.
5. Click **Create Space**.
6. Go to **Settings** → **Variables and secrets** → **New secret**:
   - Name: `GROQ_API_KEY`
   - Value: `your_groq_api_key`
7. Push code from GitHub or clone space repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/breastcancer-rag
   git push space main
   ```
8. Hugging Face will automatically build your Docker container on port `7860` and provide a free permanent public URL.

### 2. Render (Free Web Service / Blueprint)
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect repository `belalhazem511/BreastCancer-Clinical-RAG`.
3. Set Build Command: `pip install -r requirements.txt` and Start Command: `uvicorn server:app --host 0.0.0.0 --port $PORT`.
4. Add Environment Variable: `GROQ_API_KEY`.
5. Click **Deploy**.

---

## ⚖️ Clinical Disclaimer

> **IMPORTANT**: BreastCancer.ai is a clinical decision-support and research platform designed to assist healthcare professionals by summarizing and citing published **NICE Clinical Guidelines**. It does not provide definitive medical diagnoses, replace clinical judgment, or substitute for formal consultations with qualified oncology specialists. Always corroborate recommendations against the complete, official NICE guideline publications provided in the application.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
