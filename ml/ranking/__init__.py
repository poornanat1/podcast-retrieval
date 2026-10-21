"""Second-stage ranking over the hybrid candidate set.

The ranker runs PyTorch (query encoders) and LightGBM in one process. On
macOS each brings its own OpenMP runtime (PyTorch's bundled one, LightGBM's
from Homebrew, and scikit-learn's), and with more than one loaded, a
parallel region in the second runtime crashes the process: torch CPU ops
crash if LightGBM was imported first, and LightGBM crashes in training,
prediction, and even model loading if torch was imported first.

This package therefore imports torch, then LightGBM, and pins every OpenMP
runtime other than torch's to one thread, so LightGBM never starts a thread
pool. ``LIGHTGBM_THREADS`` is passed to LightGBM calls as well. One thread
costs nothing at this size: training sees tens of thousands of rows and
serving scores about 150 candidates per query. Import LightGBM only after
importing this package.
"""

import ctypes
import importlib
import sys

# Order matters (see above), so these are explicit calls rather than
# import statements an import sorter would reorder.
importlib.import_module("torch")
importlib.import_module("lightgbm")

LIGHTGBM_THREADS = 1


def _loaded_images() -> list[str]:
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    return [libc._dyld_get_image_name(i).decode()
            for i in range(libc._dyld_image_count())]


def pin_foreign_openmp() -> list[str]:
    """Set every loaded OpenMP runtime except PyTorch's to one thread.
    Returns the runtimes pinned (empty off macOS, where one runtime is
    shared)."""
    if sys.platform != "darwin":
        return []
    pinned = []
    for image in _loaded_images():
        if image.endswith("libomp.dylib") and "/torch/" not in image:
            ctypes.CDLL(image).omp_set_num_threads(1)
            pinned.append(image)
    return pinned


PINNED_OPENMP = pin_foreign_openmp()
