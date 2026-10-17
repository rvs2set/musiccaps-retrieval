"""Общие утилиты: пути и ввод-вывод."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def load_data():
    """Возвращает (corpus, queries, qrels).

    corpus:  список {"doc_id", "text"}
    queries: список {"qid", "text", "relevant": [doc_id, ...]}
    """
    corpus = read_jsonl(PROCESSED / "corpus.jsonl")
    queries = read_jsonl(PROCESSED / "queries.jsonl")
    return corpus, queries
