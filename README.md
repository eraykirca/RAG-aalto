# RAG-aalto
# RAG Application for Domain-Specific Question Answering

> Retrieval-Augmented Generation on your own documents, runnable on a
> lightweight laptop (CPU-only).

```bash
pip install -r requirements.txt
python ingest.py          # build FAISS from *.txt docs
python app.py             # interactive Q&A
```

# ✨ Features
   - Document formats:	Plain-text with markdown headers (Header\n----)
   - Embeddings:	MiniLM-L6-v2 (ingest) • BGE-Small-v1.5 (query)
   - Vector: DB	FAISS IndexFlatIP (cosine)
   - Retriever:	Top-4 cosine + top-2 header-similarity (cap = 6)
   - Generator:	google/flan-t5-large (512-token, CPU)
   - CLI: Generates answer based on relevant chunks
   - Sample corpus: verdecharge.txt, datashield_act.txt, cardiosense.txt

# 📂 Repo layout
├── data/
│   ├── verdecharge.txt          # smart-grid EV platform
│   ├── datashield_act.txt       # draft privacy legislation
│   └── cardiosense.txt          # remote cardiac monitoring
├── vectorstore/                 # auto-created FAISS index + chunks JSON
├── ingest.py                    # chunk → embed → FAISS
├── retriever.py                 # cosine + header similarity
├── app.py                       # build prompt, call Flan-T5
└── requirements.txt

# 🚀 Setup & Usage
git clone https://github.com/your-handle/rag-starter.git
cd rag-starter
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install -r requirements.txt

1. Embed and index documents
python ingest.py

2. Ask questions
python app.py
> Q: can you give an overview?
> A: VerdeCharge is a...

# 📝 Adding new documents
1. Drop a .txt file into data/ following the same header pattern:
   New Topic
   ---------
   Paragraphs…

   Subsection
   ----------
   More text…
2. Run python ingest.py again (re-embeds everything).
3. Start asking.

# ⚙️ Config knobs
| File           | Variable                          | Effect                                   |
| -------------- | --------------------------------- | ---------------------------------------- |
| `ingest.py`    | `CHUNK_SIZE`, `OVERLAP`           | granularity of chunks                    |
| `retriever.py` | `SIM_K`, `HEADER_K`, `MAX_CHUNKS` | retrieval mix & cap                      |
| `app.py`       | `safe_join(max_tokens)`           | total context tokens (≤ 512 for Flan-T5) |


