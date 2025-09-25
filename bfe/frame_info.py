"""
frame_info.py
=============
Tento modul obsahuje datovou strukturu `FrameInfo` a související utility pro uchovávání informací o snímcích.

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


@dataclass
class FrameInfo:
    idx: int
    t_sec: float
    bgr: np.ndarray
    gray: np.ndarray
    sharpness: float
    contrast: float
    exposure_score: float
    noise_score: float
    hsv_hist: np.ndarray  # (H,S,V) histogram (L2 norm)
    lowres_vec: np.ndarray  # LowRes grayscale vector (L2 norm)
    embed: np.ndarray  # concat(hsv_hist, lowres_vec) -> L2 norm
    ml_score: float = 0.0
    subscores: Optional[Dict[str, float]] = None

    # Agro proxies
    hf_energy: float = 0.0  # altitude proxy
    view_entropy_val: float = 0.0  # view proxy
    green_cover: float = 0.0  # cover proxy (0..1)
    lighting_mean: float = 0.0  # průměrná intenzita

    # Binned strata (altitude, view, cover, lighting)
    strata: Optional[Tuple[str, str, str, str]] = None
