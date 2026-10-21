import os
import sys
from pathlib import Path

# Make src/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from rtrec.pipeline import Config, run  # noqa: E402


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Train a small model once for the whole test session."""
    out = tmp_path_factory.mktemp("artifacts")
    cfg = Config(n_users=200, n_items=300, n_genres=6, n_artists=40, epochs=2, batch_size=64, candidates=50, k=10)
    run(cfg, out)
    return out
