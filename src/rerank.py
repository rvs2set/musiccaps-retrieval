"""Реранкинг кандидатов.

Два бэкенда:
  llm            — listwise-реранкинг через Claude (нужен ANTHROPIC_API_KEY)
  cross-encoder  — локальный cross-encoder (бесплатно, без ключа)

Вход: results/run_<base>.json (по умолчанию hybrid). Берём top-K кандидатов,
переупорядочиваем, хвост (после K) оставляем как есть.
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from tqdm import tqdm

from common import RESULTS, load_data, read_json, write_json

PROMPT = """Ты помогаешь музыкальному сервису подбирать треки под запрос пользователя.

Запрос пользователя:
\"\"\"{query}\"\"\"

Ниже кандидаты, каждый описан набором тегов. Упорядочи ВСЕХ кандидатов от самого подходящего запросу к наименее подходящему.

Кандидаты:
{candidates}

Ответь только JSON-массивом номеров кандидатов в порядке убывания релевантности, например [3, 1, 2]. Без пояснений."""


def parse_ranking(text, n):
    """Достаёт из ответа модели список номеров 1..n. Дубли и мусор отбрасываются."""
    m = re.search(r"\[[\d,\s]*\]", text)
    if not m:
        return []
    try:
        nums = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    seen, out = set(), []
    for x in nums:
        if isinstance(x, int) and 1 <= x <= n and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def merge_order(candidates, ranking):
    """ranking — номера (1-based) в порядке реранка; недостающие добавляем в исходном порядке."""
    ordered = [candidates[i - 1] for i in ranking]
    rest = [c for j, c in enumerate(candidates, 1) if j not in set(ranking)]
    return ordered + rest


def load_cache(path):
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r["response"]
    return cache


def rerank_llm(queries, base_run, docs, top_k, model, seed):
    import anthropic

    client = anthropic.Anthropic()  # берёт ANTHROPIC_API_KEY из окружения
    cache_path = RESULTS / "llm_cache.jsonl"
    cache = load_cache(cache_path)
    rng = random.Random(seed)
    out, n_fallback = {}, 0

    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for q in tqdm(queries, desc=f"LLM rerank ({model})"):
            cands = base_run[q["qid"]][:top_k]
            tail = base_run[q["qid"]][top_k:]
            # перемешиваем порядок показа, чтобы убрать смещение к первой позиции
            shown = cands[:]
            rng.shuffle(shown)
            listing = "\n".join(f"{i}. {docs[d]}" for i, d in enumerate(shown, 1))
            prompt = PROMPT.format(query=q["text"], candidates=listing)
            key = f"{model}|{q['qid']}|{top_k}|{seed}"
            if key in cache:
                text = cache[key]
            else:
                resp = client.messages.create(
                    model=model,
                    max_tokens=400,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                cache_f.write(json.dumps({"key": key, "response": text}) + "\n")
                cache_f.flush()
            ranking = parse_ranking(text, len(shown))
            if len(ranking) < len(shown):
                n_fallback += 1
            out[q["qid"]] = merge_order(shown, ranking) + tail
    print(f"ответов с неполным ранжированием (добито порядком показа): {n_fallback}/{len(queries)}")
    return out


def rerank_cross_encoder(queries, base_run, docs, top_k, model_name):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name)
    out = {}
    for q in tqdm(queries, desc="cross-encoder"):
        cands = base_run[q["qid"]][:top_k]
        tail = base_run[q["qid"]][top_k:]
        scores = ce.predict([(q["text"], docs[d]) for d in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        out[q["qid"]] = [cands[i] for i in order] + tail
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llm", "cross-encoder"], default="llm")
    ap.add_argument("--base", default="hybrid", help="какой run брать кандидатами: bm25 | dense | hybrid")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="только первые N запросов (для отладки)")
    args = ap.parse_args()

    corpus, queries = load_data()
    if args.limit:
        queries = queries[: args.limit]
    docs = {d["doc_id"]: d["text"] for d in corpus}
    base_run = read_json(RESULTS / f"run_{args.base}.json")

    if args.backend == "llm":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Не задан ANTHROPIC_API_KEY. Либо задайте ключ, либо используйте --backend cross-encoder")
        model = args.model or "claude-haiku-5-5"
        run = rerank_llm(queries, base_run, docs, args.top_k, model, args.seed)
        name = "llm"
    else:
        model = args.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        run = rerank_cross_encoder(queries, base_run, docs, args.top_k, model)
        name = "cross_encoder"

    write_json(RESULTS / f"run_{name}.json", run)
    print(f"сохранено results/run_{name}.json")


if __name__ == "__main__":
    main()
