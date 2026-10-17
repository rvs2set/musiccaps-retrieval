"""Baseline 2: dense retrieval (bi-encoder + FAISS) и гибрид BM25+dense через RRF."""
import argparse

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from common import RESULTS, load_data, read_json, write_json


def run_dense(corpus, queries, model_name, top_k, batch_size=64):
    model = SentenceTransformer(model_name)
    doc_emb = model.encode(
        [d["text"] for d in corpus],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    q_emb = model.encode(
        [q["text"] for q in queries],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(doc_emb.shape[1])  # косинус = скалярное произведение нормированных векторов
    index.add(doc_emb)
    _, idx = index.search(q_emb, top_k)

    doc_ids = [d["doc_id"] for d in corpus]
    return {q["qid"]: [doc_ids[i] for i in row] for q, row in zip(queries, idx)}


def rrf(runs, k=60, top_k=100):
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank)."""
    fused = {}
    for qid in runs[0]:
        scores = {}
        for run in runs:
            for rank, d in enumerate(run[qid]):
                scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank + 1)
        fused[qid] = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])][:top_k]
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    corpus, queries = load_data()
    run = run_dense(corpus, queries, args.model, args.top_k)
    write_json(RESULTS / "run_dense.json", run)
    print("сохранено results/run_dense.json")

    bm25_path = RESULTS / "run_bm25.json"
    if bm25_path.exists():
        hybrid = rrf([read_json(bm25_path), run], top_k=args.top_k)
        write_json(RESULTS / "run_hybrid.json", hybrid)
        print("сохранено results/run_hybrid.json (BM25 + dense, RRF)")
    else:
        print("run_bm25.json не найден: гибрид пропущен (сначала запустите bm25_baseline.py)")


if __name__ == "__main__":
    main()
