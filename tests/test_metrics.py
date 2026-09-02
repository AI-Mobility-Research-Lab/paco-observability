from __future__ import annotations

import pytest

from paco_observability.metrics import binary_agreement_metrics, ratio_of_sums


def test_ratio_of_sums_is_not_mean_of_frame_ratios() -> None:
    # 1/1 and 0/9: mean-of-ratios=0.5, estimand ratio=0.1.
    assert ratio_of_sums([1, 0], [1, 9]) == pytest.approx(0.1)


def test_ratio_of_sums_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        ratio_of_sums([2], [1])


def test_binary_agreement_metrics() -> None:
    metrics = binary_agreement_metrics([True, True, False, False], [True, False, True, False])
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["f1_visible"] == pytest.approx(0.5)
