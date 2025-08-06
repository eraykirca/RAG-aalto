import os, json, faiss, re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_FILE  = "data/verdecharge.txt"
VEC_DIR    = "vectorstore"
INDEX_FILE = f"{VEC_DIR}/faiss.index"
CHUNK_FILE = f"{VEC_DIR}/chunks.json"

CHUNK_SIZE = 350
OVERLAP    = 100
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------
def sectionize(md: str):
    """Yield (header, body) pairs based on 'Header\\n-----' syntax."""
    lines, header, buf = md.splitlines(), " ", []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and re.fullmatch(r"-{3,}\s*", lines[i + 1]):
            if buf:
                yield header, "\n".join(buf).strip()
                buf = []
            header = lines[i].strip()
            i += 2
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        yield header, "\n".join(buf).strip()

def main():
    os.makedirs(VEC_DIR, exist_ok=True)
    raw_text = open(DATA_FILE, encoding="utf-8").read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
    )

    #header prepended to every chunk
    chunks = []
    for hdr, body in sectionize(raw_text):
        for piece in splitter.split_text(body):
            chunks.append(f"{hdr}: {piece}")

    # embed + FAISS
    enc  = SentenceTransformer(EMB_MODEL)
    vecs = enc.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, INDEX_FILE)
    json.dump(chunks, open(CHUNK_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"Indexed {len(chunks)} header‑tagged chunks → {INDEX_FILE}")

if __name__ == "__main__":
    main()
