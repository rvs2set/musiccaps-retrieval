"""Baseline 1: BM25 по лексическому совпадению."""
import argparse

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from common import RESULTS, load_data, tokenize, write_json


def run_bm25(corpus, queries, top_k):
    bm25 = BM25Okapi([tokenize(d["text"]) for d in corpus])
    doc_ids = [d["doc_id"] for d in corpus]
    run = {}
    for q in tqdm(queries, desc="BM25"):
        scores = bm25.get_scores(tokenize(q["text"]))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        run[q["qid"]] = [doc_ids[i] for i in order]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()
    corpus, queries = load_data()
    run = run_bm25(corpus, queries, args.top_k)
    write_json(RESULTS / "run_bm25.json", run)
    print("сохранено results/run_bm25.json")


if __name__ == "__main__":
    main()
