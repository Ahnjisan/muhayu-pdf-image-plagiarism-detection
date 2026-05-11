from src.model.threshold import find_best_threshold_by_youden


def test_threshold():
    th, tpr, fpr = find_best_threshold_by_youden([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 <= th <= 0.9
