import numpy as np

from rtrec import data


def test_generate_shapes_and_ranges():
    ds = data.generate(n_users=50, n_items=80, n_genres=4, n_artists=10, min_events=5, max_events=9, seed=1)
    assert len(ds.items) == 80
    assert ds.interactions["user_idx"].between(0, 49).all()
    assert ds.interactions["item_idx"].between(0, 79).all()
    assert ds.items["genre"].between(0, 3).all()
    assert ds.items["artist"].between(0, 9).all()


def test_generate_is_deterministic():
    a = data.generate(n_users=20, n_items=40, min_events=5, max_events=10, seed=7).interactions
    b = data.generate(n_users=20, n_items=40, min_events=5, max_events=10, seed=7).interactions
    assert a.equals(b)


def test_each_user_has_no_repeated_items():
    ds = data.generate(n_users=30, n_items=60, min_events=5, max_events=10, seed=2)
    dupes = ds.interactions.duplicated(["user_idx", "item_idx"]).sum()
    assert dupes == 0


def test_leave_two_out_has_one_validation_and_one_test_per_user():
    ds = data.generate(n_users=40, n_items=80, min_events=5, max_events=10, seed=3)
    train, validation, test = data.leave_two_out(ds.interactions)
    users = set(ds.interactions["user_idx"])
    assert set(validation["user_idx"]) == users
    assert set(test["user_idx"]) == users
    assert len(validation) == len(users) and len(test) == len(users)


def test_leave_two_out_is_leakage_free():
    ds = data.generate(n_users=40, n_items=80, min_events=5, max_events=10, seed=4)
    train, validation, test = data.leave_two_out(ds.interactions)
    for user, held_out in test.groupby("user_idx"):
        later = ds.interactions[ds.interactions["user_idx"] == user]
        assert held_out["ts"].iloc[0] == later["ts"].max()
        assert held_out["item_idx"].iloc[0] not in set(train[train["user_idx"] == user]["item_idx"])
    assert np.isin(test["item_idx"], train["item_idx"]).sum() <= len(test)
    assert len(train) + len(validation) + len(test) == len(ds.interactions)
