"""
__init__.py
===========
Tento soubor inicializuje balíček `bfe` a zpřístupňuje jeho moduly.
"""

from .frame_info import FrameInfo
from .quality_metrics import (
    variance_of_laplacian,
    estimate_contrast,
    exposure_metrics,
    exposure_score_from_metrics,
    estimate_noise_score,
)
from .embeddings import (
    hsv_histogram,
    lowres_embedding,
    combined_embed,
    cosine_similarity,
)
from .proxies import (
    altitude_proxy,
    view_entropy,
    green_cover_ratio,
    classify_lighting,
)
from .stratification import AgroStratifier
from .deduplication import (
    auto_eps_from_adjacent_sims,
    deduplicate_quality_first,
    deduplicate_dbscan,
)
