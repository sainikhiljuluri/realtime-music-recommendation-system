"""Music4All-Onion loader: fixed-seed stratified user sample, streamed in chunks.

Source: Music4All-Onion (Moscati et al., CIKM 2022), CC-BY-4.0, Zenodo record 15394646.
Only the listening log (userid_trackid_timestamp) and the track genre TF-IDF table are used.
No artist metadata is used. The dataset is not stored in this repository.

Memory: the listening log is about 253M rows. It is read in chunks. Pass 1 counts events
per user. Pass 2 keeps only the sampled users, so peak memory stays near the size of the
sample rather than the full log.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 5_000_000


@dataclass(frozen=True)
class MusicSample:
    user_ids: np.ndarray       # original user ids, sorted, length n_users
    item_ids: np.ndarray       # track ids, length n_items (dense item index = position)
    item_genre: np.ndarray     # primary genre index per item, -1 when no genre weight
    events: pd.DataFrame       # columns: user_idx, item_idx, ts (int64 seconds), chronological per user
    n_genres: int


def read_track_genres(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Track ids and each track's primary genre (highest TF-IDF weight)."""
    table = pd.read_csv(path, sep="\t", compression="bz2", index_col=0)
    weights = table.to_numpy(dtype=np.float32)
    primary = weights.argmax(axis=1).astype(np.int64)
    primary[weights.max(axis=1) <= 0] = -1
    return table.index.to_numpy(dtype=object), primary, table.shape[1]


def count_user_events(log_path: Path) -> pd.Series:
    """Pass 1: number of listening events per user, streamed."""
    total = None
    for chunk in pd.read_csv(log_path, sep="\t", compression="bz2", usecols=["user_id"], chunksize=CHUNK):
        counts = chunk["user_id"].value_counts()
        total = counts if total is None else total.add(counts, fill_value=0)
    return total.astype(np.int64).sort_index()


def stratified_user_sample(counts: pd.Series, n_users: int, min_events: int, n_strata: int, seed: int) -> np.ndarray:
    """Fixed-seed sample with equal shares from quantile strata of listening activity."""
    eligible = counts[counts >= min_events]
    if n_users > len(eligible):
        raise ValueError(f"only {len(eligible)} users have at least {min_events} events")
    edges = np.quantile(eligible.to_numpy(), np.linspace(0, 1, n_strata + 1))
    strata = np.clip(np.searchsorted(edges, eligible.to_numpy(), side="right") - 1, 0, n_strata - 1)
    rng = np.random.default_rng(seed)
    per = np.full(n_strata, n_users // n_strata)
    per[: n_users - per.sum()] += 1
    chosen = []
    for s in range(n_strata):
        members = eligible.index.to_numpy()[strata == s]
        take = min(per[s], len(members))
        chosen.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def load_sample(data_dir: Path, n_users: int = 10_000, min_events: int = 50, n_strata: int = 10,
                seed: int = 0) -> MusicSample:
    log_path = data_dir / "userid_trackid_timestamp.tsv.bz2"
    genre_path = data_dir / "id_genres_tf-idf.tsv.bz2"
    counts = count_user_events(log_path)
    users = stratified_user_sample(counts, n_users, min_events, n_strata, seed)
    track_ids, item_genre, n_genres = read_track_genres(genre_path)
    track_index = pd.Index(track_ids)
    user_index = pd.Index(users)

    parts = []
    for chunk in pd.read_csv(log_path, sep="\t", compression="bz2", chunksize=CHUNK,
                             dtype={"user_id": "int64", "track_id": "string", "timestamp": "string"}):
        keep = chunk["user_id"].isin(user_index)
        chunk = chunk[keep]
        if chunk.empty:
            continue
        item = track_index.get_indexer(chunk["track_id"].astype(object))
        valid = item >= 0
        chunk = chunk[valid]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["timestamp"], format="%Y-%m-%d %H:%M:%S").astype("int64") // 10**9
        parts.append(pd.DataFrame({
            "user_idx": user_index.get_indexer(chunk["user_id"]).astype(np.int64),
            "item_idx": item[valid].astype(np.int64),
            "ts": ts.to_numpy(dtype=np.int64),
        }))
    events = pd.concat(parts, ignore_index=True).sort_values(["user_idx", "ts"], kind="stable").reset_index(drop=True)
    return MusicSample(users, track_ids, item_genre, events, n_genres)


def save_sample(sample: MusicSample, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, user_ids=sample.user_ids, item_ids=sample.item_ids.astype(str),
                        item_genre=sample.item_genre, n_genres=np.array(sample.n_genres),
                        user_idx=sample.events["user_idx"].to_numpy(), item_idx=sample.events["item_idx"].to_numpy(),
                        ts=sample.events["ts"].to_numpy())


def load_saved(path: Path) -> MusicSample:
    z = np.load(path, allow_pickle=False)
    events = pd.DataFrame({"user_idx": z["user_idx"], "item_idx": z["item_idx"], "ts": z["ts"]})
    return MusicSample(z["user_ids"], z["item_ids"].astype(object), z["item_genre"], events, int(z["n_genres"]))


def summary(sample: MusicSample) -> dict:
    ev = sample.events
    return {
        "users": int(len(sample.user_ids)),
        "tracks_in_catalog": int(len(sample.item_ids)),
        "tracks_with_events": int(ev["item_idx"].nunique()),
        "events": int(len(ev)),
        "events_per_user_median": float(ev.groupby("user_idx").size().median()),
        "genres": int(sample.n_genres),
        "tracks_without_genre": int((sample.item_genre < 0).sum()),
        "time_range_utc": [pd.to_datetime(int(ev["ts"].min()), unit="s").isoformat(),
                           pd.to_datetime(int(ev["ts"].max()), unit="s").isoformat()],
    }
