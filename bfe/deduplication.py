"""
deduplication.py
================
Tento modul obsahuje funkce pro deduplikaci snímků na základě kosinové podobnosti a dalších metod.

"""

from typing import List, Optional, Dict
from sklearn.cluster import DBSCAN
import numpy as np
from bfe.frame_info import FrameInfo
from bfe.embeddings import cosine_similarity


def auto_eps_from_adjacent_sims(
    frames: List[FrameInfo],
    quantile: Optional[float] = None,
    config: Optional[Dict] = None,
) -> float:
    """
    Automaticky vypočítá hodnotu eps pro DBSCAN na základě sousedních podobností.
    """
    if len(frames) < 3:
        return 0.10
    constants_config = (config or {}).get("constants", {})
    q = (
        quantile
        if quantile is not None
        else constants_config.get("dbscan_auto_eps_quantile", 0.90)
    )
    bounds = constants_config.get("dbscan_eps_bounds", [0.02, 0.30])
    sims = []
    for i in range(len(frames) - 1):
        sims.append(cosine_similarity(frames[i].embed, frames[i + 1].embed))
    Q = float(np.quantile(np.array(sims, dtype=np.float32), q))
    eps = max(float(bounds[0]), min(float(bounds[1]), 1.0 - Q))
    return eps


def deduplicate_quality_first(
    frames: List[FrameInfo],
    cosine_threshold: Optional[float] = None,
    config: Optional[Dict] = None,
) -> List[FrameInfo]:
    """
    Jednoduchá greedy deduplikace na základě kosinové podobnosti.
    """
    if cosine_threshold is None:
        deduplication_config = (config or {}).get("deduplication", {})
        cosine_threshold = deduplication_config.get("cosine_threshold", 0.85)
    if not frames:
        return []

    selected: List[FrameInfo] = []
    for f in sorted(frames, key=lambda x: x.ml_score, reverse=True):
        is_duplicate = False
        for s in selected:
            if cosine_similarity(f.embed, s.embed) > cosine_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(f)

    return selected


def deduplicate_dbscan(
    frames: List[FrameInfo],
    eps: Optional[float] = None,
    min_samples: int = 1,
    config: Optional[Dict] = None,
) -> List[FrameInfo]:
    """
    Deduplikace pomocí DBSCAN na embeddingových vektorech.
    """
    if not frames:
        return []

    dedup_conf = (config or {}).get("deduplication", {})
    if eps is None:
        try:
            eps = dedup_conf.get("eps", None)
            if eps is None:
                eps = auto_eps_from_adjacent_sims(frames, config=config)
        except Exception:
            eps = 0.1

    X = np.vstack([f.embed for f in frames]).astype(np.float32)

    db = DBSCAN(eps=eps, min_samples=int(min_samples), metric="cosine")
    labels = db.fit_predict(X)

    clusters: Dict[int, List[FrameInfo]] = {}
    for f, lab in zip(frames, labels):
        clusters.setdefault(int(lab), []).append(f)

    selected: List[FrameInfo] = []
    for lab, items in clusters.items():
        best = max(items, key=lambda t: t.ml_score)
        selected.append(best)

    selected = sorted(selected, key=lambda x: x.ml_score, reverse=True)
    return selected
