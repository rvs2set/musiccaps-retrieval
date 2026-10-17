"""Считает метрики для всех results/run_*.json и пишет таблицу results/results.md."""
from common import RESULTS, load_data, read_json
from metrics import bootstrap_ci, evaluate_run, per_query_scores

NAMES = {
    "bm25": "BM25",
    "dense": "Dense (MiniLM + FAISS)",
    "hybrid": "Hybrid (RRF)",
    "cross_encoder": "Hybrid → Cross-encoder",
    "llm": "Hybrid → LLM rerank",
}
ORDER = ["bm25", "dense", "hybrid", "cross_encoder", "llm"]


def main():
    _, queries = load_data()
    qrels = {q["qid"]: q["relevant"] for q in queries}

    rows = []
    for key in ORDER:
        path = RESULTS / f"run_{key}.json"
        if not path.exists():
            continue
        run = read_json(path)
        m = evaluate_run(run, qrels)
        ndcgs = list(per_query_scores(run, qrels, 10).values())
        lo, hi = bootstrap_ci(ndcgs)
        rows.append((NAMES[key], m, (lo, hi)))

    if not rows:
        raise SystemExit("Нет файлов results/run_*.json")

    header = "| Метод | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@10 (95% CI) | MRR@10 | N |\n"
    header += "|---|---|---|---|---|---|---|---|\n"
    lines = []
    for name, m, (lo, hi) in rows:
        lines.append(
            f"| {name} | {m['Recall@1']:.3f} | {m['Recall@5']:.3f} | {m['Recall@10']:.3f} | "
            f"{m['Recall@20']:.3f} | {m['NDCG@10']:.3f} [{lo:.3f}, {hi:.3f}] | {m['MRR@10']:.3f} | {m['n_queries']} |"
        )
    table = header + "\n".join(lines) + "\n"
    (RESULTS / "results.md").write_text(table, encoding="utf-8")
    print(table)
    print("сохранено results/results.md")


if __name__ == "__main__":
    main()
