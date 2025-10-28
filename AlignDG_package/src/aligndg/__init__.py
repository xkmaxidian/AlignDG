__version__ = "1.0.2"

from .utils import *
from .graph import *
from .uopt import *
import os, warnings
from .utils.seed import set_global_seed, seed_worker


_disable = os.getenv("AlignDG_DISABLE_AUTO_SEED", "0") == "1"
if not _disable:
    try:
        seed = int(os.getenv("AlignDG_SEED", "42"))
        det  = bool(int(os.getenv("AlignDG_DETERMINISTIC", "1")))
        set_global_seed(seed, deterministic=det)
    except Exception as e:
        warnings.warn(f"[AlignDG] auto seed failed, ignored: {e}")
