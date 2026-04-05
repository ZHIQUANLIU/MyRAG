# RAG Application

Retrieval Augmented Generation application using LangChain + Google Gemini.

## Supported Formats

- PDF
- TXT
- DOCX / DOC
- XLSX / XLS
- EPUB

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key:
Edit `.env` file and set your Google AI Studio API key:
```
GOOGLE_API_KEY=your_api_key_here
```
Get free API key at: https://aistudio.google.com/app/apikey

3. Put documents in `my_pdfs/` folder

4. Run:
```bash
python rag_app.py
```

## Configuration

Edit `rag_app.py` to customize:
- `CHUNK_SIZE`: Text chunk size (default: 500)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)
- `TOP_K`: Number of retrieved documents (default: 3)
- `VECTOR_STORE_PATH`: Vector database path (default: faiss_index)

## Features

- Multi-format document support (PDF, TXT, DOCX, XLSX, EPUB)
- Local FAISS vector database (persisted for reuse)
- Interactive Q&A
- Rolling log files (20MB, 3 backups)

## Commands

- `quit` / `exit` / `q`: Exit
- `sources`: View retrieved source documents

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies