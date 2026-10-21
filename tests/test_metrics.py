import math

from rtrec.metrics import ndcg_at_k, recall_at_k


def test_recall_hit_and_miss():
    assert recall_at_k([4, 9, 2], target=9, k=2) == 1.0
    assert recall_at_k([4, 9, 2], target=2, k=2) == 0.0


def test_ndcg_rewards_higher_ranks():
    assert ndcg_at_k([7, 8, 9], target=7, k=3) == 1.0
    assert math.isclose(ndcg_at_k([7, 8, 9], target=9, k=3), 1 / math.log2(4))
    assert ndcg_at_k([7, 8, 9], target=9, k=2) == 0.0
