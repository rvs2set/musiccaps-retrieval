"""Метрики ранжирования с бинарной релевантностью."""
import math


def recall_at_k(ranked, relevant, k):
    """Доля релевантных документов, попавших в топ-k."""
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked, relevant, k):
    rel = set(relevant)
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(run, qrels, ks=(1, 5, 10, 20)):
    """run: {qid: [doc_id,...]}, qrels: {qid: [doc_id,...]} -> средние метрики."""
    out = {}
    qids = [q for q in qrels if q in run]
    n = len(qids)
    if n == 0:
        raise ValueError("Нет пересечения запросов в run и qrels")
    for k in ks:
        out[f"Recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in qids) / n
    out["NDCG@10"] = sum(ndcg_at_k(run[q], qrels[q], 10) for q in qids) / n
    out["MRR@10"] = sum(mrr_at_k(run[q], qrels[q], 10) for q in qids) / n
    out["n_queries"] = n
    return out


def per_query_scores(run, qrels, k=10):
    """NDCG@k по каждому запросу (для бутстрэпа и анализа ошибок)."""
    return {q: ndcg_at_k(run[q], qrels[q], k) for q in qrels if q in run}


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Доверительный интервал среднего через бутстрэп."""
    import random

    rng = random.Random(seed)
    vals = list(values)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi
