import pytest

from rtrec.experiments import assign, bucket


def test_assignment_is_deterministic():
    assert assign("exp1", "u5") == assign("exp1", "u5")


def test_assignment_is_independent_across_experiments():
    users = [f"u{i}" for i in range(2000)]
    same = sum(assign("a", u) == assign("b", u) for u in users) / len(users)
    assert 0.4 < same < 0.6


def test_split_is_close_to_requested_share():
    users = [f"u{i}" for i in range(20000)]
    share = sum(assign("exp", u, 0.2) == "treatment" for u in users) / len(users)
    assert abs(share - 0.2) < 0.01


def test_bucket_range_and_invalid_share():
    assert 0 <= bucket("e", "u1") < 10_000
    with pytest.raises(ValueError):
        assign("e", "u1", treatment_share=1.5)
