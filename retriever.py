import faiss, json, re
from sentence_transformers import SentenceTransformer
from collections import defaultdict

INDEX_FILE = "vectorstore/faiss.index"
CHUNK_FILE = "vectorstore/chunks.json"
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

SIM_K      = 4   #similarity slices
HEADER_K   = 2   #number of headers to pull
MAX_CHUNKS = 6   # absolute cap sent to the LLM

class Retriever:
    def __init__(self):
        self.enc    = SentenceTransformer(EMB_MODEL)
        self.index  = faiss.read_index(INDEX_FILE)
        self.chunks = json.load(open(CHUNK_FILE, encoding="utf-8"))

        # build header: indices map and embed each unique header
        hdr_to_idx = defaultdict(list)
        for i, ch in enumerate(self.chunks):
            hdr_to_idx[self._header_of(ch)].append(i)

        self.hdr_texts   = list(hdr_to_idx.keys())
        self.hdr_indices = list(hdr_to_idx.values())

        hdr_embs = self.enc.encode(self.hdr_texts, convert_to_numpy=True)
        faiss.normalize_L2(hdr_embs)
        self.hdr_index = faiss.IndexFlatIP(hdr_embs.shape[1])
        self.hdr_index.add(hdr_embs)

    @staticmethod
    def _header_of(chunk: str) -> str:
        return chunk.split(":", 1)[0].strip().lower()

    def get_chunks(self, question: str):
        # cosine similarity top-SIM_K
        q_vec = self.enc.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        _, sim_idx = self.index.search(q_vec, SIM_K)
        sim_idx = sim_idx[0].tolist()

        # header similarity top-HEADER_K
        _, hdr_idx = self.hdr_index.search(q_vec, HEADER_K)
        hdr_idx = hdr_idx[0].tolist()

        header_chunks = []
        for hi in hdr_idx:
            header_chunks.extend(self.hdr_indices[hi])

        # merge with cap
        final, seen = [], set()
        for i in header_chunks + sim_idx:
            if i not in seen:
                final.append(i)
                seen.add(i)
            if len(final) == MAX_CHUNKS:        # stop at hard cap
                break

        return [self.chunks[i] for i in final]
