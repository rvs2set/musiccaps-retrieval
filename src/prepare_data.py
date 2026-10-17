"""Скачивает MusicCaps с Hugging Face и готовит corpus.jsonl / queries.jsonl.

Постановка:
  * документ (трек)  = список аспектов-тегов (aspect_list), напр. "pop, soft female vocal, sad"
  * запрос           = свободное текстовое описание музыки (caption)
  * релевантный      = трек, к которому относится это описание

Аудио не нужно: работаем только с текстом.
"""
import argparse
import ast
import random

from common import PROCESSED, write_jsonl


def parse_aspects(raw):
    if isinstance(raw, list):
        return raw
    try:
        return list(ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return [a.strip() for a in str(raw).strip("[]").split(",")]


def first_sentence(text):
    for sep in (". ", "! ", "? "):
        if sep in text:
            return text.split(sep)[0].strip() + "."
    return text.strip()


def build(rows, n_queries, query_mode, seed):
    corpus, candidates = [], []
    for r in rows:
        aspects = parse_aspects(r["aspect_list"])
        # убираем дубли аспектов, сохраняя порядок
        aspects = list(dict.fromkeys(a.strip() for a in aspects if a.strip()))
        if not aspects or not str(r["caption"]).strip():
            continue
        doc_id = r["ytid"] + f"_{int(r['start_s'])}"
        corpus.append({"doc_id": doc_id, "text": ", ".join(aspects)})
        q = str(r["caption"]).strip()
        if query_mode == "first_sentence":
            q = first_sentence(q)
        candidates.append({"text": q, "relevant": [doc_id]})

    rng = random.Random(seed)
    idx = list(range(len(candidates)))
    rng.shuffle(idx)
    idx = idx[: min(n_queries, len(idx))]
    queries = [
        {"qid": f"q{i}", "text": candidates[j]["text"], "relevant": candidates[j]["relevant"]}
        for i, j in enumerate(sorted(idx))
    ]
    return corpus, queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--query-mode", choices=["caption", "first_sentence"], default="first_sentence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="локальный CSV MusicCaps (если Hugging Face недоступен)")
    args = ap.parse_args()

    if args.csv:
        import csv

        with open(args.csv, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        from datasets import load_dataset

        rows = list(load_dataset("google/MusicCaps", split="train"))
    corpus, queries = build(rows, args.n_queries, args.query_mode, args.seed)

    write_jsonl(PROCESSED / "corpus.jsonl", corpus)
    write_jsonl(PROCESSED / "queries.jsonl", queries)
    print(f"corpus: {len(corpus)} треков, queries: {len(queries)} (mode={args.query_mode})")
    print("пример запроса:", queries[0]["text"])
    print("релевантный док:", next(c["text"] for c in corpus if c["doc_id"] == queries[0]["relevant"][0]))


if __name__ == "__main__":
    main()
