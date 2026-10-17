"""Показывает примеры запросов, где метод ошибся (для раздела «Выводы» в README)."""
import argparse

from common import RESULTS, load_data, read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hybrid", help="bm25 | dense | hybrid | cross_encoder | llm")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=int, default=10, help="ошибка = релевантный трек не в топ-k")
    args = ap.parse_args()

    corpus, queries = load_data()
    docs = {d["doc_id"]: d["text"] for d in corpus}
    run = read_json(RESULTS / f"run_{args.run}.json")

    shown = 0
    for q in queries:
        ranked = run.get(q["qid"])
        if not ranked:
            continue
        rel = q["relevant"][0]
        pos = ranked.index(rel) + 1 if rel in ranked else None
        if pos is None or pos > args.k:
            print(f"\nЗАПРОС: {q['text']}")
            print(f"  правильный (позиция {pos}): {docs[rel]}")
            print(f"  выдано на 1 месте:         {docs[ranked[0]]}")
            shown += 1
            if shown >= args.n:
                break
    if shown == 0:
        print("ошибок не найдено")


if __name__ == "__main__":
    main()
