import pandas as pd

from rtrec.movielens import load


def _write_fixture(root):
    pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 2, 3],
        "movieId": [10, 20, 30, 10, 20, 40, 50, 10],
        "rating": [5.0, 4.0, 3.0, 4.5, 5.0, 4.0, 2.0, 5.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
    }).to_csv(root / "ratings.csv", index=False)
    pd.DataFrame({
        "movieId": [10, 20, 30, 40, 50],
        "title": ["a", "b", "c", "d", "e"],
        "genres": ["Drama|Comedy", "Action", "Comedy", "Drama", "(no genres listed)"],
    }).to_csv(root / "movies.csv", index=False)


def test_loader_keeps_positives_and_drops_short_users(tmp_path):
    _write_fixture(tmp_path)
    ds = load(tmp_path, min_rating=4.0, min_events=3)
    # User 1 has two positives and user 3 has one, so both are dropped. User 2 keeps its three
    # ratings >= 4.0; its rating of 2.0 is excluded.
    assert ds.n_users == 1
    assert len(ds.interactions) == 3
    assert set(ds.interactions["ts"]) == {4, 5, 6}


def test_loader_uses_primary_genre_and_dense_ids(tmp_path):
    _write_fixture(tmp_path)
    ds = load(tmp_path, min_rating=4.0, min_events=2)
    assert ds.interactions["user_idx"].max() < ds.n_users
    assert ds.interactions["item_idx"].max() < ds.n_items
    assert ds.n_artists == 1
    assert ds.items["artist"].eq(0).all()
    assert ds.n_genres == len(set(ds.items["genre"]))
