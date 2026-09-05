# 🎓 Teaching Assistant — AI-Powered RAG System

An AI teaching assistant that lets you **upload a PDF, audio, or video file**, and then **ask questions about its content** in natural language (Arabic or English). It uses **Retrieval-Augmented Generation (RAG)** with a local **Ollama** LLM, so no data leaves your machine.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat"/>
  <img src="https://img.shields.io/badge/FAISS-4169E1?style=flat"/>
  <img src="https://img.shields.io/badge/Whisper-412991?style=flat&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat&logo=ollama&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=flat&logo=gradio&logoColor=white"/>
</p>

---

## ✨ Features

- 📄 **PDF Upload** — extracts text automatically (via PyMuPDF)
- 🎥 **Audio/Video Upload** — transcribes speech to text automatically (via Whisper + ffmpeg)
- 🔍 **RAG Pipeline** — chunks the document, embeds it (Sentence-Transformers), stores it in a FAISS vector index, and retrieves the most relevant chunks for each question
- 🧠 **LangGraph Workflow** — classifies whether a question is relevant to the uploaded document before answering, and rejects off-topic questions gracefully
- 🌐 **Bilingual** — answers in the same language as the question (Arabic or English)
- 🖥️ **Simple UI** — Gradio interface with separate tabs for direct chat, PDF upload, media upload, and document Q&A

## 🏗️ Architecture

```
┌─────────────┐      HTTP       ┌──────────────┐
│  Gradio UI   │ ───────────── ▶│   FastAPI     │
│ (frontend.py)│ ◀───────────── │ (backend/main)│
└─────────────┘                 └──────┬───────┘
                                        │
                     ┌──────────────────┼───────────────────┐
                     ▼                  ▼                   ▼
              PyMuPDF (PDF)     Whisper + ffmpeg      LangGraph RAG
              text extraction   (audio/video → text)  (classify → retrieve → answer)
                                                              │
                                                              ▼
                                                    FAISS + Sentence-Transformers
                                                              │
                                                              ▼
                                                     Ollama (llama3.2, local LLM)
```

## 📁 Project Structure

```
teaching-assistant/
├── backend/
│   ├── main.py          # FastAPI app: routes for /ask, /upload, /upload_media, /ask_rag
│   ├── rag.py           # Chunking, embeddings, FAISS vector store, similarity search
│   └── rag_graph.py     # LangGraph workflow: classify → retrieve → answer / reject
├── frontend.py           # Gradio UI (chat, file upload, RAG Q&A)
├── tests/                 # Test & evaluation scripts
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup Guide (from scratch)

### 1. Prerequisites

- **Python 3.10+**
- **Git**
- **[Ollama](https://ollama.com/download)** installed locally (used to run the LLM)
- **ffmpeg** is bundled automatically via the `imageio-ffmpeg` package — no manual install needed

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/teaching-assistant.git
cd teaching-assistant
```

### 3. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install PyTorch (CPU build)

`torch` is pinned to a CPU build, which isn't on the default PyPI index, so install it separately first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install the rest of the requirements

```bash
pip install -r requirements.txt
```

### 6. Set up environment variables

Copy the example file and adjust if needed:

```bash
cp .env.example .env      # macOS/Linux
copy .env.example .env     # Windows
```

```
OLLAMA_URL=http://localhost:11434/api/generate
MODEL_NAME=llama3.2
```

### 7. Pull and run the Ollama model

In a separate terminal:

```bash
ollama pull llama3.2
ollama serve
```

Keep this terminal running in the background — the backend calls it at `http://localhost:11434`.

### 8. Run the backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 8000
```

### 9. Run the frontend (Gradio)

In a **new terminal** (with the venv activated again):

```bash
python frontend.py
```

Gradio will print a local URL (usually `http://127.0.0.1:7860`) — open it in your browser.

---

## 🧪 Running Tests

```bash
pytest tests/
```

> Note: some scripts in `tests/` are demonstration scripts rather than strict `assert`-based tests — check each file before relying on `pytest`'s pass/fail output.

## 🚦 API Endpoints

| Method | Endpoint         | Description                                  |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | Health check                                  |
| POST   | `/ask`           | Ask the LLM directly (no document context)    |
| POST   | `/upload`        | Upload a PDF and build its vector index       |
| POST   | `/upload_media`  | Upload audio/video, transcribe, and index it  |
| POST   | `/ask_rag`       | Ask a question about the last uploaded file   |

## ⚠️ Known Limitations

- Single global session — only one document can be active at a time; uploading a new file replaces the previous one. Not yet suitable for multiple concurrent users.
- Requires a locally running Ollama instance; there's no cloud-LLM fallback yet.
- Whisper transcription defaults to Arabic (`language="ar"`); change this in `backend/main.py` for other languages.

## 🗺️ Roadmap

- [ ] Per-user/session document isolation
- [ ] Support for multiple simultaneous documents
- [ ] Streaming responses
- [ ] Dockerfile for one-command setup

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
