"""
embeddings.py
=============
Tento modul obsahuje funkce pro výpočet embeddingů snímků, včetně HSV histogramů, low-res embeddingů a kombinovaných embeddingů.

"""

from typing import Tuple, Optional, Dict
import numpy as np
import cv2


def hsv_histogram(
    bgr: np.ndarray, bins: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Vypočítá HSV histogram snímku.
    """
    if bins is None:
        bins = (16, 16, 16)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = bins
    h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    hist = np.concatenate([h.ravel(), s.ravel(), v.ravel()]).astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-9
    return hist


def lowres_embedding(
    bgr: np.ndarray, size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Vytvoří low-res embedding snímku.
    """
    if size is None:
        size = (64, 36)
    small = cv2.resize(bgr, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= gray.mean()
    vec = gray.reshape(-1)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.astype(np.float32)


def combined_embed(
    bgr: np.ndarray, config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vytvoří kombinovaný embedding snímku (HSV histogram + low-res embedding).
    """
    embedding_config = (config or {}).get("embedding", {})
    hsv_bins = embedding_config.get("hsv_bins", (16, 16, 16))
    lowres_size = embedding_config.get("lowres_size", (64, 36))

    hsv = hsv_histogram(bgr, bins=hsv_bins)
    low = lowres_embedding(bgr, size=lowres_size)
    emb = np.concatenate([hsv, low]).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    return hsv, low, emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Vypočítá kosinovou podobnost mezi dvěma vektory.
    """
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))
