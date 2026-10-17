import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

from metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k
from prepare_data import build, first_sentence, parse_aspects
from rerank import merge_order, parse_ranking


def test_metrics_single_relevant():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c"], 2) == 0.0
    assert recall_at_k(ranked, ["c"], 3) == 1.0
    assert mrr_at_k(ranked, ["c"], 10) == 1 / 3
    assert math.isclose(ndcg_at_k(ranked, ["c"], 10), 1 / math.log2(4))
    assert ndcg_at_k(ranked, ["a"], 10) == 1.0
    assert ndcg_at_k(ranked, ["z"], 10) == 0.0


def test_evaluate_run():
    run = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": ["a"], "q2": ["y"]}
    m = evaluate_run(run, qrels, ks=(1, 2))
    assert m["Recall@1"] == 0.5
    assert m["Recall@2"] == 1.0


def test_parse_ranking_robust():
    assert parse_ranking("Вот ответ: [3, 1, 2]", 3) == [3, 1, 2]
    assert parse_ranking("[1, 1, 2, 9, 0]", 3) == [1, 2]  # дубли и выход за диапазон отброшены
    assert parse_ranking("нет списка", 3) == []


def test_merge_order_fills_missing():
    assert merge_order(["a", "b", "c"], [3]) == ["c", "a", "b"]
    assert merge_order(["a", "b", "c"], []) == ["a", "b", "c"]


def test_prepare_build():
    rows = [
        {"ytid": "x1", "start_s": 30, "aspect_list": "['pop', 'sad', 'pop']", "caption": "A sad pop song. It is slow."},
        {"ytid": "x2", "start_s": 0, "aspect_list": "['rock']", "caption": "Loud rock."},
        {"ytid": "x3", "start_s": 0, "aspect_list": "[]", "caption": "empty tags"},
    ]
    corpus, queries = build(rows, n_queries=10, query_mode="first_sentence", seed=1)
    assert len(corpus) == 2  # строка без тегов отброшена
    assert corpus[0]["text"] == "pop, sad"  # дубли убраны
    assert {q["text"] for q in queries} == {"A sad pop song.", "Loud rock."}
    assert parse_aspects("['a', 'b']") == ["a", "b"]
    assert first_sentence("One. Two.") == "One."
