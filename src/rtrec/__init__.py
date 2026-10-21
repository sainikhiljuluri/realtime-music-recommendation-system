"""Real-time music recommendation: two-tower retrieval, FAISS, LightGBM ranking, Redis cache."""
import os

# On macOS, the pip wheels for torch, faiss, and lightgbm each load an OpenMP runtime.
# Running them multi-threaded in one process crashes. Everything runs single-threaded
# here. The flag lets duplicate runtimes load. It is an unsupported workaround, so tests
# verify FAISS results against brute-force numpy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
