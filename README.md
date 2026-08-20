---
title: BreastCancer.ai Clinical RAG
emoji: 🩺
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Grounded NICE Guideline Clinical Decision Support with Groq LPU & Voice AI
---

# 🩺 BreastCancer.ai — Clinical RAG Decision-Support Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Groq Cloud](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed%20ONNX%20%28%3C50MB%29-FF6F61?style=for-the-badge)](https://github.com/qdrant/fastembed)
[![BM25](https://img.shields.io/badge/Search-BM25%20Okapi-0052CC?style=for-the-badge)](https://github.com/dorianbrown/rank_bm25)
[![Docker](https://img.shields.io/badge/Docker-Production%20Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br />

[![Latency](https://img.shields.io/badge/Inference_Speed-%3C0.8s%20(Groq%20LPU)-blueviolet?style=flat-square)](https://groq.com/)
[![RAM](https://img.shields.io/badge/Memory_Footprint-%3C50MB%20RAM-success?style=flat-square)](https://github.com/qdrant/fastembed)
[![Guidelines](https://img.shields.io/badge/Clinical_Standard-NICE%20Guidelines%20(UK)-red?style=flat-square)](https://www.nice.org.uk/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<br />

**An evidence-grounded clinical AI decision-support platform designed for breast oncology, strictly adhering to National Institute for Health and Care Excellence (NICE) guidelines.**

<br />

<p align="center">
  <a href="https://huggingface.co/new-space?template=belalhazem511/BreastCancer-Clinical-RAG"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/deploy-to-spaces-lg.svg" alt="Deploy to Hugging Face Spaces"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://render.com/deploy?repo=https://github.com/belalhazem511/BreastCancer-Clinical-RAG"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
</p>

<br />

[✨ Key Innovations](#-key-innovations) • [📚 Knowledge Base](#-indexed-nice-guidelines-knowledge-base) • [🏗️ Architecture](#️-system-architecture) • [🔬 Mathematical Formula](#-mathematical-hybrid-scoring-formulation) • [💡 Clinical Scenarios](#-interactive-clinical-query-scenarios) • [🎙️ Voice AI & Orb](#️-voice-assistant--interactive-glowing-orb) • [🚀 Quickstart](#-quickstart-guide) • [📡 API Reference](#-api-reference) • [☁️ Free Cloud Deploy](#️-cloud-deployment--100-free-hosting)

</div>

---

## 📖 Executive Summary

**BreastCancer.ai** is an end-to-end, production-ready clinical AI decision-support platform built for oncologists, multidisciplinary cancer teams (MDTs), and medical researchers. It bridges the ultra-fast reasoning of **Groq LPUs** with strict, deterministic medical grounding from official **NICE Clinical Guidelines**:

* 📘 **NICE NG101**: Early and locally advanced breast cancer: diagnosis and management
* 📙 **NICE CG81**: Advanced breast cancer: diagnosis and treatment
* 📗 **NICE CG164**: Familial breast cancer: classification, care and managing risk

### 🎯 The Clinical Problem It Solves
Generic Large Language Models frequently generate medical hallucinations, cite outdated drug regimens, or cannot provide precise document-level accountability. **BreastCancer.ai** enforces a **zero-unsupported-assertions** policy using hybrid vector-lexical retrieval, dynamic threshold gating, page-level PDF preview synchronization, and two-way voice dictation.

---

## 📚 Indexed NICE Guidelines Knowledge Base

| Guideline Code | Official Publication Title | Clinical Scope & Key Chapters | Indexed Chunks | Total Pages | Direct PDF Access |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **NICE NG101** | *Early and locally advanced breast cancer: diagnosis and management* | Staging, endocrine therapy (tamoxifen/aromatase inhibitors), adjuvant chemotherapy, HER2+ targeted biologics, radiotherapy | **215 chunks** | 67 pages | [`NG101.pdf`](file:///B:/hakthon3%20-%20Copy%20%282%29/RAG%20system/Data/NG101.pdf) |
| **NICE CG81** | *Advanced breast cancer: diagnosis and treatment* | Metastatic disease, visceral crises, bisphosphonates for bone metastases, endocrine sequencing, palliative regimens | **108 chunks** | 44 pages | [`CG81.pdf`](file:///B:/hakthon3%20-%20Copy%20%282%29/RAG%20system/Data/CG81.pdf) |
| **NICE CG164** | *Familial breast cancer: classification, care and managing risk* | BRCA1/BRCA2 genetic testing, lifetime risk stratification, annual MRI/mammography surveillance, chemoprevention | **103 chunks** | 52 pages | [`CG164.pdf`](file:///B:/hakthon3%20-%20Copy%20%282%29/RAG%20system/Data/CG164.pdf) |

---

## ✨ Key Innovations

| Capability | Description |
| :--- | :--- |
| **🔍 Hybrid Dual-Stage Retrieval** | Combines **Dense Vector Search** (`BAAI/bge-small-en-v1.5` ONNX) with **Sparse Lexical Search** (Rank-BM25 Okapi) using min-max weighted fusion (`0.65 Dense + 0.35 Sparse`). |
| **🛡️ Anti-Hallucination Guardrails** | Strict similarity thresholding ($S_{min} \ge 0.58$) rejects out-of-domain queries automatically with transparent low-confidence disclaimers. |
| **⚡ Ultra-Low Memory Engine** | Replaced heavy PyTorch runtimes (~550 MB RAM) with **FastEmbed ONNX Runtime** (<50 MB RAM), enabling flawless deployment on free 512 MB cloud tiers. |
| **📑 Verifiable Evidence Drawer** | Synchronized side-drawer displays exact guideline excerpt, section code, page bounds, and provides one-click deep links directly into the original PDF. |
| **🎙️ Real-Time Voice Dictation** | Hands-free clinical query input powered by the Web Speech API with smart silence detection and automatic query dispatch. |
| **🔊 Text-to-Speech Voice Assistant** | Reads out clinical summaries and evidence recommendations with animated soundwave feedback and click-to-stop controls. |
| **🔮 Interactive Reactive Orb** | Central UI orb dynamically responds to voice states: idle breathing, pulsing pink listening waves, and emerald processing vortex swirls. |
| **⚡ Groq Multi-Model Failover Chain** | Instant sub-second inference with resilient failover across multiple model tiers (`gpt-oss-120b`, `gpt-oss-20b`, `compound-mini`, `qwen-27b`, `allam-7b`). |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph UI ["Clinical Frontend Interface"]
        Clinician["Clinician / Oncologist"] -->|Voice Dictation or Text Query| Composer["Clinical Workspace Composer"]
        Composer -->|Voice State Active| GlowingOrb["Reactive Glowing Orb (Listening / Processing)"]
        Composer -->|POST /api/chat| Gateway["FastAPI REST API Gateway"]
        Gateway -->|Stream Response| AnswerCard["Evidence Answer Card"]
        AnswerCard -->|Read Aloud| VoiceAssistant["Voice Assistant (Speech Synthesis)"]
        AnswerCard -->|Inspect Citation| EvidenceDrawer["Interactive Evidence Drawer"]
        EvidenceDrawer -->|Deep Link page=N| PDFServer["PDF Streamer (/api/pdf/{doc})"]
    end

    subgraph RAG ["Hybrid RAG Engine (Retrieval.py)"]
        Gateway --> Preprocess["Query Normalization & Tokenization"]
        Preprocess --> DenseSearch["Dense Embedding Search\n(BAAI/bge-small-en-v1.5 ONNX)"]
        Preprocess --> SparseSearch["Lexical Sparse Search\n(BM25 Okapi)"]
        DenseSearch --> Normalization["Min-Max Score Normalization"]
        SparseSearch --> Normalization
        Normalization --> ScoreFusion["Hybrid Weighted Fusion\n(0.65 Dense + 0.35 Sparse)"]
        ScoreFusion --> TopChunks["Top-K Reranked Guidelines Chunks"]
    end

    subgraph LLM ["Grounded Medical Inference"]
        TopChunks --> ConfidenceGate{"Score Threshold\n(Score >= 0.58?)"}
        ConfidenceGate -->|"No (Score < 0.58)"| SafeFallback["Anti-Hallucination Safe Fallback\n(Confidence: Low, Zero Outside Knowledge)"]
        ConfidenceGate -->|"Yes (Score >= 0.58)"| GroundingPrompt["NICE Clinical Grounding Prompt"]
        GroundingPrompt --> GroqEngine["Groq Fast LPU Inference\n(Multi-Model Resilient Failover)"]
        GroqEngine --> StructuredParser["Structured JSON Response Parser"]
        StructuredParser --> AnswerPayload["Structured Recommendations + Exact Citations"]
    end

    AnswerPayload --> AnswerCard
    SafeFallback --> AnswerCard
```

---

## 🔬 Mathematical Hybrid Scoring Formulation

The retrieval engine employs a mathematical fusion model combining geometric dense semantic representation with statistical lexical matching:

### 1. Score Normalization & Fusion
For a user query $q$ and guideline candidate chunk $d$:

$$\text{Score}_{\text{dense}}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}, \quad \text{Score}_{\text{bm25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

$$\text{Score}_{\text{hybrid}}(q, d) = w_{\text{dense}} \cdot \left(\frac{\text{Score}_{\text{dense}} - \min(S_d)}{\max(S_d) - \min(S_d) + \epsilon}\right) + w_{\text{bm25}} \cdot \left(\frac{\text{Score}_{\text{bm25}} - \min(S_b)}{\max(S_b) - \min(S_b) + \epsilon}\right)$$

> **Hyperparameters**: $w_{\text{dense}} = 0.65$, $w_{\text{bm25}} = 0.35$, $k_1 = 1.5$, $b = 0.75$.

### 2. Deterministic Anti-Hallucination Confidence Gating
$$\text{Decision Gate}(q) = \begin{cases} 
\text{Synthesize Grounded Clinical Answer via Groq LPU}, & \text{if } \max_{d} \text{Score}_{\text{hybrid}}(q, d) \ge 0.58 \\
\text{Reject Query with Low-Confidence Safe Fallback}, & \text{if } \max_{d} \text{Score}_{\text{hybrid}}(q, d) < 0.58
\end{cases}$$

---

## 💡 Interactive Clinical Query Scenarios

<details open>
<summary><b>💊 Scenario 1: Hormone-Receptor Positive (ER+) Early Breast Cancer</b> <i>(Click to collapse/expand)</i></summary>

> **Clinical Query**: *"What is the recommended endocrine therapy for ER-positive early breast cancer in premenopausal vs postmenopausal patients?"*  
>
> **Grounded NICE Recommendations**:
> 1. **Premenopausal Patients**: Offer **Tamoxifen** as the standard initial adjuvant endocrine therapy (`NICE NG101 Section 1.11.1`).
> 2. **Postmenopausal Patients (High Risk)**: Offer an **Aromatase Inhibitor** (anastrozole or letrozole) as initial adjuvant therapy for women at medium or high risk of disease recurrence (`NICE NG101 Section 1.11.4`).
> 3. **Duration**: Continue adjuvant endocrine therapy for a minimum of 5 years, with consideration of extended therapy up to 10 years based on risk scoring (`NICE NG101 Section 1.11.7`).
>
> **Verified Reference**: [NICE NG101 Section 1.11 (Pages 32–33)](file:///B:/hakthon3%20-%20Copy%20%282%29/RAG%20system/Data/NG101.pdf) • **Confidence**: `High (Score: 0.83)`
</details>

<details>
<summary><b>🧬 Scenario 2: High-Risk Familial Surveillance & BRCA Mutation Carriers</b> <i>(Click to expand)</i></summary>

> **Clinical Query**: *"What annual surveillance protocol is recommended for confirmed BRCA1 or BRCA2 mutation carriers?"*  
>
> **Grounded NICE Recommendations**:
> 1. **MRI Surveillance (Ages 30–49)**: Offer annual **magnetic resonance imaging (MRI)** surveillance to women aged 30–49 who possess a verified BRCA1 or BRCA2 gene mutation (`NICE CG164 Section 1.6.4`).
> 2. **Mammography (Ages 40–69)**: Offer annual **mammographic surveillance** to women aged 40–69 with high familial risk or BRCA mutation status (`NICE CG164 Section 1.6.8`).
> 3. **Chemoprevention**: Discuss risk-reducing medications (tamoxifen or raloxifene for postmenopausal women without a history of thromboembolism) (`NICE CG164 Section 1.9.2`).
>
> **Verified Reference**: [NICE CG164 Section 1.6 (Pages 24–26)](file:///B:/hakthon3%20-%20Copy%20%282%29/RAG%20system/Data/CG164.pdf) • **Confidence**: `High (Score: 0.75)`
</details>

<details>
<summary><b>🦴 Scenario 3: Advanced Breast Cancer & Bone Metastases</b> <i>(Click to expand)</i></summary>

> **Clinical Query**: *"What bisphosphonate therapy is recommended for advanced breast cancer patients with bone metastases?"*  
>
> **Grounded NICE Recommendations**:
> 1. **Bisphosphonate Initiation**: Offer **bisphosphonates** (such as zoledronic acid, pamidronate disodium, or sodium clodronate) to all patients newly diagnosed with bone metastases to reduce skeletal-related events and alleviate bone pain (`NICE CG81 Section 1.5.1`).
> 2. **Analgesic Synergy**: Combine bisphosphonates with appropriate analgesic protocols and assess need for palliative radiotherapy for localized intractable pain (`NICE CG81 Section 1.5.3`).
>
> **Verified Reference**: [NICE CG81 Section 1.5 (Pages 19–21)](file:///B:/hakthon3%20-%20Copy%20%282%29/RAG%20system/Data/CG81.pdf) • **Confidence**: `High (Score: 0.79)`
</details>

---

## 🎙️ Voice Assistant & Interactive Glowing Orb

BreastCancer.ai includes a state-of-the-art multimodal clinical interface designed for sterile or busy clinical environments where hands-free interaction is essential:

```
                  ┌──────────────────────────────────────────────┐
                  │          🔮 Interactive Glowing Orb           │
                  │   • Idle: Soft ambient breathing rings       │
                  │   • Listening: Vibrant pink pulsing waves    │
                  │   • Processing: High-speed emerald vortex    │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    🎙️ Hands-Free Voice Input                      🔊 Clinical Voice Output
    • Click Mic OR Click Orb to Speak             • Automatic speech synthesis
    • Real-time speech-to-text transcription      • Clear, natural medical pronunciation
    • Smart 1.6s silence auto-submission          • Interactive 🔊 Listen / ⏹ Stop badge
```

---

## 📁 Repository Structure

```
BreastCancer-Clinical-RAG/
├── server.py                   # FastAPI application server & REST endpoints
├── run.py                      # One-click launcher (port management & auto-browser)
├── clean_port.py               # Port-conflict management & clean shutdown utility
├── run.bat                     # Windows one-click batch launcher
├── verify_full_cycle.py        # Automated end-to-end verification test suite
├── requirements.txt            # Python production dependencies (FastEmbed, FastAPI, Groq)
├── Dockerfile                  # Multi-stage production container configuration
├── docker-compose.yml          # Container orchestration configuration
├── render.yaml                 # 1-Click Render Cloud deployment blueprint
├── Procfile                    # PaaS process configuration
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git exclusion rules (strictly ignores secrets)
│
├── .github/
│   └── workflows/
│       └── verify.yml          # GitHub Actions Continuous Integration workflow
│
├── RAG system/                 # Core RAG engine & knowledge base
│   ├── Retrieval.py            # Hybrid retrieval, ONNX vector encoding & scoring
│   ├── Chunking_Metadata.py    # Guideline parser & page-aware chunking pipeline
│   ├── Parsing_Cleaning.py     # PDF text extraction & structure normalization
│   ├── Embedding.py            # BGE vector generation script
│   ├── requirements.txt        # RAG-specific requirements
│   └── Data/
│       ├── NG101.pdf           # NICE Early & Locally Advanced Breast Cancer (550 KB)
│       ├── CG81.pdf            # NICE Advanced Breast Cancer Guideline (198 KB)
│       ├── CG164.pdf           # NICE Familial Breast Cancer Guideline (255 KB)
│       ├── chunks_metadata.json# Pre-computed chunk metadata with section & page bounds
│       └── chunk_vectors.npy   # Pre-computed 384-dimensional BGE embedding vectors
│
└── breast-cancer-ai-clean/     # Frontend Web Interface
    ├── index.html              # Animated splash entry screen
    ├── home.html               # Clinical landing page & interactive orb
    ├── chat.html               # Clinical workspace & Evidence Drawer
    ├── uploaded-pdfs.html      # Guideline library catalog & inline PDF viewer
    ├── citation-history.html   # Evidence citation history log & markdown export
    ├── assets/
    │   └── logo.svg            # Vector UI branding icon
    ├── css/
    │   ├── base.css            # Design tokens, typography & pulsating mic animations
    │   ├── splash.css          # Splash screen animations
    │   ├── home.css            # Landing layout & interactive orb voice animations
    │   ├── chat.css            # Chat interface, soundwave bars & evidence drawer
    │   └── library.css         # Document grid & preview modal
    └── js/
        ├── splash.js           # Splash loader logic
        ├── home.js             # Orb voice interaction & query submission
        ├── chat.js             # RAG integration, Speech-to-Text & Speech Synthesis
        ├── library.js          # Document viewer & PDF modal controller
```

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.11+** installed ([Download Python](https://www.python.org/downloads/))
- A free **Groq Cloud API Key** ([Get free key here](https://console.groq.com/keys))

### 1. Clone the Repository
```bash
git clone https://github.com/belalhazem511/BreastCancer-Clinical-RAG.git
cd BreastCancer-Clinical-RAG
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
PORT=8000
```

### 5. Launch the Application

#### Option A: One-Click Python Launcher (Recommended)
```bash
python run.py
```
> *Automatically verifies port availability, starts the FastAPI server, and launches your default browser at `http://127.0.0.1:8000`.*

#### Option B: Windows Batch Launcher
Double click `run.bat` or run in terminal:
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

### 1. `POST /api/chat`
Execute a clinical query grounded against NICE guidelines.

**Request Payload:**
```json
{
  "question": "What is the recommended endocrine therapy for hormone receptor positive early breast cancer?"
}
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "has_context": true,
  "confidence": "High",
  "source_match": "83%",
  "summary": "Offer adjuvant endocrine therapy to all people with ER-positive invasive early breast cancer...",
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

### 2. `GET /api/sources`
Retrieve metadata and status for all connected clinical guidelines.

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

### 3. `GET /api/pdf/{filename}`
Streams the guideline PDF with inline viewing headers.
- **Example**: `GET /api/pdf/NG101.pdf`
- **Jump directly to page**: `/api/pdf/NG101.pdf#page=32`

---

### 4. `GET /api/health`
System health check endpoint.
```json
{
  "status": "healthy",
  "engine": "FastEmbed ONNX + BM25",
  "memory_footprint": "<50MB",
  "groq_connected": true,
  "indexed_chunks": 426
}
```

---

## 🧪 Automated Verification Suite

The repository includes a comprehensive verification test suite validating all REST routes, static assets, PDF serving, hybrid RAG scoring, and out-of-context rejection:

```bash
python verify_full_cycle.py
```

### Test Suite Execution Output:
```
============================================================
  RUNNING FULL-CYCLE VERIFICATION TEST SUITE
============================================================
[OK] Static HTML Page /                         -> 200 OK
[OK] Static HTML Page /home.html                -> 200 OK
[OK] Static HTML Page /chat.html                -> 200 OK
[OK] Static HTML Page /uploaded-pdfs.html       -> 200 OK
[OK] Static HTML Page /citation-history.html    -> 200 OK
[OK] Static Asset     /js/chat.js               -> 200 OK
[OK] Static Asset     /css/chat.css             -> 200 OK
[OK] API /api/health            -> 200 OK (Status: healthy)
[OK] API /api/sources           -> 200 OK (3 connected guidelines: NG101, CG81, CG164)
[OK] PDF Serving /api/pdf/NG101.pdf    -> 200 OK (550,639 bytes)
[OK] PDF Serving /api/pdf/CG81.pdf     -> 200 OK (198,127 bytes)
[OK] PDF Serving /api/pdf/CG164.pdf    -> 200 OK (255,533 bytes)

[*] Testing Clinical Query (NG101): 'What is the recommended endocrine therapy for ER+ early breast cancer?'
[OK] Clinical Query RAG Response -> 200 OK (Confidence: High, Citations: 4 chunks)

[*] Testing Clinical Query (CG164): 'What surveillance is recommended for BRCA mutation carriers?'
[OK] Familial Breast Cancer RAG Response -> 200 OK (Confidence: High, Citations: 4 chunks)

[*] Testing Out-of-Context Query (Threshold Rejection): 'What is the capital of Australia?'
[OK] Out-of-Context Rejection   -> 200 OK (Confidence: Low, 0 hallucinations)

============================================================
  ALL FULL-CYCLE VERIFICATION TESTS PASSED SUCCESSFULLY!  
============================================================
```

---

## ☁️ Cloud Deployment — 100% Free Hosting

### 1. Hugging Face Spaces (100% Free CPU Docker)
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Space Name: `breastcancer-rag`
3. Space SDK: **Docker** (Blank)
4. Visibility: **Public**
5. Go to **Settings** → **Variables and secrets** → **New secret**:
   - `GROQ_API_KEY`: `your_groq_api_key`
6. Push code to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/breastcancer-rag
   git push space main
   ```
7. Hugging Face builds your container and serves it on port `7860`.

---

### 2. Render (100% Free Web Service)
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Web Service**.
2. Connect your GitHub repository: `belalhazem511/BreastCancer-Clinical-RAG`.
3. Configure Build Settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variable:
   - `GROQ_API_KEY`: `your_groq_api_key`
5. Click **Deploy Web Service**.

---

## ⚖️ Clinical Safety Disclaimer

> [!IMPORTANT]
> **BreastCancer.ai** is a clinical decision-support and research assistant designed to assist qualified healthcare professionals by indexing, retrieving, and citing published **NICE Clinical Guidelines**. It does not constitute medical advice, provide autonomous clinical diagnoses, or replace professional oncology judgment. Clinicians should corroborate all generated recommendations against the primary NICE guideline publications directly accessible within the application.

---

## 📄 License

This project is open-source software licensed under the **[MIT License](LICENSE)**.

<div align="center">
  <sub>Built with clinical rigor for the Oncology AI Community.</sub>
</div>
