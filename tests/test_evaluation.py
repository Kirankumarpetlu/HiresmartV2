from scripts.eval_evaluator import metrics
from scripts.eval_retrieval import recall


def test_recall_and_level_metrics():
    assert recall(["a", "b", "c"], ["b", "c"], 2) == 0.5
    assert recall([], [], 10) == 0
    assert metrics([(3, 3), (2, 1)]) == {"compared": 2, "agreement_rate": 0.5, "mean_absolute_error": 0.5}
