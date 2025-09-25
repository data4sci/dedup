#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-Driven Frame Curation — Agro Stratification (MVP)
====================================================
Funkční kurace snímků z agro-droních videí se *stratifikací* podle:
  - altitude proxy (HF energy),
  - view angle proxy (entropie orientací),
  - vegetation cover (ExG/VARI proxy),
  - lighting (průměrná intenzita).

Pipeline:
  1) Pre-filter (ostrost/kontrast) + rychlé embeddingy (HSV+LowRes).
  2) ML skóre (quality + content_novelty).
  3) Výběr podle target_size (stratifikace) nebo threshold (top % kandidátů).
  4) Quality-aware deduplikace (greedy s cosine threshold).
  5) Uložení snímků + manifest.

Závislosti:
    pip install opencv-python numpy scikit-learn pyyaml

Použití:
    python balanced_frame_extractor.py input.mp4 -o out_dir --config config.yaml
"""
from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Iterable
from collections import defaultdict

import numpy as np
import cv2
import yaml
import logging
import time

logger = logging.getLogger(__name__)

from sklearn.cluster import DBSCAN


# -----------------------------------------------------------------------------
# Datové struktury
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Nízké level metriky kvality
# -----------------------------------------------------------------------------
def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


def exposure_metrics(
    gray: np.ndarray, config: Optional[Dict] = None
) -> Tuple[float, float, float]:
    image_proc_config = (config or {}).get("image_processing", {})
    exposure_config = image_proc_config.get("exposure", {})
    constants_config = (config or {}).get("constants", {})

    underexposure_bins = exposure_config.get("underexposure_bins", 10)
    overexposure_start = exposure_config.get("overexposure_start", 246)
    epsilon = constants_config.get("epsilon", 1e-9)

    mean = float(np.mean(gray))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + epsilon)
    under = float(hist[:underexposure_bins].sum())
    over = float(hist[overexposure_start:].sum())
    return mean, under, over


def exposure_score_from_metrics(
    mean: float, under_frac: float, over_frac: float, config: Optional[Dict] = None
) -> float:
    image_proc_config = (config or {}).get("image_processing", {})
    exposure_config = image_proc_config.get("exposure", {})

    center_value = exposure_config.get("center_value", 128.0)
    center_penalty_weight = exposure_config.get("center_penalty_weight", 0.6)
    clip_penalty_weight = exposure_config.get("clip_penalty_weight", 0.4)
    clip_multiplier = exposure_config.get("clip_multiplier", 2.0)

    center_penalty = abs(mean - center_value) / center_value
    clip_penalty = clip_multiplier * (under_frac + over_frac)
    raw = 1.0 - min(
        1.0, center_penalty_weight * center_penalty + clip_penalty_weight * clip_penalty
    )
    return float(max(0.0, min(1.0, raw)))


def estimate_noise_score(gray: np.ndarray, config: Optional[Dict] = None) -> float:
    image_proc_config = (config or {}).get("image_processing", {})

    gaussian_kernel_size = tuple(image_proc_config.get("gaussian_kernel_size", [3, 3]))
    gaussian_sigma = image_proc_config.get("gaussian_sigma", 0)
    noise_scaling_factor = image_proc_config.get("noise_scaling_factor", 25.0)

    blur = cv2.GaussianBlur(gray, gaussian_kernel_size, gaussian_sigma)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    resid_std = float(np.std(resid))
    score = 1.0 - min(1.0, resid_std / noise_scaling_factor)
    return float(max(0.0, score))


# -----------------------------------------------------------------------------
# Rychlé embeddingy
# -----------------------------------------------------------------------------
def hsv_histogram(
    bgr: np.ndarray, bins: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
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
    embedding_config = (config or {}).get("embedding", {})
    hsv_bins = embedding_config.get("hsv_bins", (16, 16, 16))
    lowres_size = embedding_config.get("lowres_size", (64, 36))

    hsv = hsv_histogram(bgr, bins=hsv_bins)
    low = lowres_embedding(bgr, size=lowres_size)
    emb = np.concatenate([hsv, low]).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    return hsv, low, emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# -----------------------------------------------------------------------------
# Agro proxies a binning
# -----------------------------------------------------------------------------
def altitude_proxy(gray: np.ndarray, config: Optional[Dict] = None) -> float:
    # HF energie jako průměr absolutní hodnoty high-pass
    image_proc_config = (config or {}).get("image_processing", {})

    gaussian_kernel_size = tuple(image_proc_config.get("gaussian_kernel_size", [3, 3]))
    gaussian_sigma = image_proc_config.get("gaussian_sigma", 0)

    hp = gray.astype(np.float32) - cv2.GaussianBlur(
        gray, gaussian_kernel_size, gaussian_sigma
    )
    return float(np.mean(np.abs(hp)))


def view_entropy(
    gray: np.ndarray, bins: Optional[int] = None, config: Optional[Dict] = None
) -> float:
    if bins is None:
        proxies_config = (config or {}).get("proxies", {})
        bins = proxies_config.get("view_entropy_bins", 8)

    image_proc_config = (config or {}).get("image_processing", {})
    constants_config = (config or {}).get("constants", {})

    sobel_kernel_size = image_proc_config.get("sobel_kernel_size", 3)
    epsilon = constants_config.get("epsilon", 1e-9)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    hist, _ = np.histogram(
        ang.ravel(), bins=bins, range=(0, 2 * np.pi), weights=mag.ravel()
    )
    p = hist / (hist.sum() + epsilon)
    ent = -np.sum(p * np.log(p + epsilon))
    return float(ent)


def green_cover_ratio(
    bgr: np.ndarray, threshold: Optional[float] = None, config: Optional[Dict] = None
) -> float:
    # Excess Green proxy (0..1)
    if threshold is None:
        threshold = 0.6
    small_eps = (config or {}).get("constants", {}).get("epsilon_small", 1e-6)
    b, g, r = cv2.split(bgr.astype(np.float32) + small_eps)
    exg = 2 * g - r - b
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-9)
    return float(np.mean(exg_norm >= threshold))


def classify_lighting(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def bin_altitude(hf: float, q50: float) -> str:
    if hf >= q50:
        return "low"  # hodně detailů -> nízko
    return "high"  # málo detailů -> vysoko


def bin_view(ent: float, t: Optional[float] = None) -> str:
    if t is None:
        t = 1.8
    if ent >= t:
        return "nadir"
    return "oblique"


def bin_cover(ratio: float, threshold: float = 0.5) -> str:
    if ratio >= threshold:
        return "dense"
    return "sparse"


def bin_lighting(mean_int: float, threshold: Optional[float] = None) -> str:
    if threshold is None:
        threshold = 115
    if mean_int < threshold:
        return "dark"
    return "bright"


# -----------------------------------------------------------------------------
# Streamování videa + prefilter + výpočet proxy
# -----------------------------------------------------------------------------
def iter_video_frames(
    video_path: str, stride: int = 1, config: Optional[Dict] = None
) -> Iterable[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít video: {video_path}")
    constants_config = (config or {}).get("constants", {})
    fps = cap.get(cv2.CAP_PROP_FPS) or constants_config.get("fallback_fps", 30.0)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, (idx / fps), frame
        idx += 1
    cap.release()


def prefilter_candidates(
    video_path: str,
    stride: int = 1,
    min_sharpness: float = 80.0,
    min_contrast: float = 20.0,
    config: Optional[Dict] = None,
) -> List[FrameInfo]:
    logger.debug(
        f"Starting pre-filter with stride={stride}, min_sharpness={min_sharpness}, min_contrast={min_contrast}"
    )
    out: List[FrameInfo] = []
    total_frames = 0
    filtered_out = 0

    output_config = (config or {}).get("output", {})
    log_intervals = output_config.get("log_intervals", {})
    frames_processed_interval = log_intervals.get("frames_processed", 1000)

    for idx, t_sec, bgr in iter_video_frames(video_path, stride=stride, config=config):
        total_frames += 1
        if total_frames % frames_processed_interval == 0:
            logger.debug(f"Processed {total_frames} frames, kept {len(out)} candidates")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(gray)
        contr = estimate_contrast(gray)

        if sharp < min_sharpness or contr < min_contrast:
            filtered_out += 1
            logger.debug(
                f"Frame {idx} filtered out: sharpness={sharp:.1f} (min={min_sharpness}), contrast={contr:.1f} (min={min_contrast})"
            )
            continue

        mean, under, over = exposure_metrics(gray, config=config)
        expo = exposure_score_from_metrics(mean, under, over, config=config)
        noise = estimate_noise_score(gray, config=config)
        hsv, low, emb = combined_embed(bgr, config=config)

        # agro proxies (continuous)
        proxies_config = (config or {}).get("proxies", {})
        view_bins = proxies_config.get("view_entropy_bins", 8)
        green_threshold = proxies_config.get("green_cover_threshold", 0.6)

        hf = altitude_proxy(gray, config=config)
        ent = view_entropy(gray, bins=view_bins, config=config)
        gcr = green_cover_ratio(bgr, threshold=green_threshold, config=config)
        light_mean = float(np.mean(gray))

        logger.debug(
            f"Frame {idx}: sharp={sharp:.1f}, contrast={contr:.1f}, expo={expo:.3f}, hf={hf:.2f}, ent={ent:.2f}, green={gcr:.3f}"
        )

        out.append(
            FrameInfo(
                idx=idx,
                t_sec=t_sec,
                bgr=bgr,
                gray=gray,
                sharpness=sharp,
                contrast=contr,
                exposure_score=expo,
                noise_score=noise,
                hsv_hist=hsv,
                lowres_vec=low,
                embed=emb,
                hf_energy=hf,
                view_entropy_val=ent,
                green_cover=gcr,
                lighting_mean=light_mean,
            )
        )

    logger.info(
        f"Pre-filter complete: {total_frames} frames processed, {len(out)} candidates kept, {filtered_out} filtered out"
    )
    return out


# -----------------------------------------------------------------------------
# ML skórování (bez DL)
# -----------------------------------------------------------------------------
class MLFrameScorer:
    def __init__(
        self,
        novelty_memory: int = 64,
        novelty_threshold: float = 0.3,
        config: Optional[Dict] = None,
    ):
        self.prototypes: List[np.ndarray] = []
        self.novelty_memory = int(novelty_memory)
        self.novelty_threshold = novelty_threshold

        # Load scoring weights from config
        scoring_config = (config or {}).get("scoring", {})
        weights = scoring_config.get("weights", {})
        quality_components = scoring_config.get("quality_components", {})
        scale_ranges = scoring_config.get("scale_ranges", {})

        # Default weights if not in config
        self.weight_quality = weights.get("quality", 0.5)
        self.weight_novelty = weights.get("content_novelty", 0.5)

        self.quality_sharpness = quality_components.get("sharpness", 0.35)
        self.quality_contrast = quality_components.get("contrast", 0.30)
        self.quality_exposure = quality_components.get("exposure", 0.25)
        self.quality_noise = quality_components.get("noise", 0.10)

        self.novelty_memory = scoring_config.get("novelty_memory", 64)
        self.sharpness_range = scale_ranges.get("sharpness", [80.0, 300.0])
        self.contrast_range = scale_ranges.get("contrast", [20.0, 80.0])

    @staticmethod
    def _scale(x: float, lo: float, hi: float) -> float:
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo + 1e-9))))

    def _quality_score(self, f: FrameInfo) -> float:
        s = self._scale(f.sharpness, self.sharpness_range[0], self.sharpness_range[1])
        c = self._scale(f.contrast, self.contrast_range[0], self.contrast_range[1])
        e = f.exposure_score
        n = f.noise_score
        return (
            self.quality_sharpness * s
            + self.quality_contrast * c
            + self.quality_exposure * e
            + self.quality_noise * n
        )

    def _geom_score(self, f: FrameInfo) -> float:
        gx = cv2.Sobel(f.gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f.gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
        bins = 8
        hist, _ = np.histogram(
            ang.ravel(), bins=bins, range=(0, 2 * np.pi), weights=mag.ravel()
        )
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
        geom = 1.0 - float(hist.max())
        return float(max(0.0, min(1.0, geom)))

    def _novelty_score(self, emb: np.ndarray) -> float:
        if not self.prototypes:
            return 1.0
        sims = [
            cosine_similarity(emb, p) for p in self.prototypes[-self.novelty_memory :]
        ]
        max_sim = max(sims) if sims else 0.0
        return float(max(0.0, min(1.0, 1.0 - max_sim)))

    def score(self, frames: List[FrameInfo], config: Optional[Dict] = None) -> None:
        output_config = (config or {}).get("output", {})
        log_intervals = output_config.get("log_intervals", {})
        ml_scoring_interval = log_intervals.get("ml_scoring", 500)

        logger.debug(f"Starting ML scoring for {len(frames)} frames")
        for i, f in enumerate(frames):
            if i % ml_scoring_interval == 0 and i > 0:
                logger.debug(f"ML scoring progress: {i}/{len(frames)} frames")

            q = self._quality_score(f)
            nov = self._novelty_score(f.embed)
            total = self.weight_quality * q + self.weight_novelty * nov
            f.ml_score = float(total)
            f.subscores = {"quality": q, "content_novelty": nov}

            logger.debug(
                f"Frame {f.idx}: quality={q:.3f}, novelty={nov:.3f} → ML_score={total:.3f}"
            )

            if nov > self.novelty_threshold:
                self.prototypes.append(f.embed)
                logger.debug(f"Frame {f.idx} added to prototypes (novelty={nov:.3f})")

        logger.info(
            f"ML scoring complete. Prototypes collected: {len(self.prototypes)}"
        )


# -----------------------------------------------------------------------------
# Stratifikace (4 osy) + YAML cíle a limity
# -----------------------------------------------------------------------------
class AgroStratifier:
    """
    Převádí kontinuální proxy -> binned straty a udržuje výběr podle YAML targetů.
    YAML formát (viz config.yaml):
      stratification:
        axes:
          altitude: [low, high]
          view: [nadir, oblique]
          cover: [sparse, dense]
          lighting: [dark, bright]
        targets:
          "altitude:low|view:nadir|cover:dense|lighting:bright": 0.25
          "*": 0.75
        limits:
          windy_max_ratio: 0.15
          sparse_max_ratio: 0.30
    """

    def __init__(self, config: Dict):
        sconf = (config or {}).get("stratification", {})
        self.axes = sconf.get(
            "axes",
            {
                "altitude": ["low", "high"],
                "view": ["nadir", "oblique"],
                "cover": ["sparse", "dense"],
                "lighting": ["dark", "bright"],
            },
        )
        self.targets_raw: Dict[str, float] = sconf.get("targets", {"*": 1.0})
        self.limits = sconf.get(
            "limits", {"windy_max_ratio": 0.15, "sparse_max_ratio": 0.30}
        )
        self.counts: Dict[str, int] = defaultdict(int)
        self.total_selected: int = 0

        # Expand '*' later proportionally over missing combinations
        self.combinations = self._all_combinations()

        # Compile explicit targets and distribute wildcard
        self.targets = self._compile_targets()

    def _all_combinations(self) -> List[str]:
        from itertools import product

        keys = list(self.axes.keys())
        values = [self.axes[k] for k in keys]
        combos = []
        for vals in product(*values):
            parts = [f"{k}:{v}" for k, v in zip(keys, vals)]
            combos.append("|".join(parts))
        return combos

    def _compile_targets(self) -> Dict[str, float]:
        explicit_total = sum(v for k, v in self.targets_raw.items() if k != "*")
        wildcard = self.targets_raw.get("*", 0.0)
        remaining = max(0.0, 1.0 - explicit_total)
        if wildcard > 0:
            # normalize wildcard to remaining
            wildcard_share = remaining
        else:
            wildcard_share = 0.0

        # find combos without explicit target
        explicit_keys = {k for k in self.targets_raw.keys() if k != "*"}
        missing = [c for c in self.combinations if c not in explicit_keys]
        per = (wildcard_share / len(missing)) if missing and wildcard_share > 0 else 0.0

        targets = {}
        for c in self.combinations:
            if c in self.targets_raw and c != "*":
                targets[c] = float(self.targets_raw[c])
            else:
                targets[c] = float(per)
        # normalize to sum=1 (small numeric drift)
        s = sum(targets.values()) + 1e-9
        for k in list(targets.keys()):
            targets[k] = targets[k] / s
        return targets

    @staticmethod
    def combo_key(altitude: str, view: str, cover: str, lighting: str) -> str:
        return f"altitude:{altitude}|view:{view}|cover:{cover}|lighting:{lighting}"

    def select(
        self,
        frames_sorted: List[FrameInfo],
        target_size: Optional[int],
        config: Optional[Dict] = None,
    ) -> List[FrameInfo]:
        selection_config = (config or {}).get("selection", {})
        stratified_split_ratio = selection_config.get("stratified_split_ratio", 0.5)
        high_quality_threshold = selection_config.get("high_quality_threshold", 0.95)

        selected: List[FrameInfo] = []
        for f in frames_sorted:
            if target_size is not None and len(selected) >= target_size:
                break

            # derive key and ratios
            a, v, c, l = f.strata  # type: ignore
            key = self.combo_key(a, v, c, l)

            # enforce limits (example: sparse cover cap)
            if c == "sparse":
                if self._current_ratio(selected, cover="sparse") >= self.limits.get(
                    "sparse_max_ratio", 1.0
                ):
                    continue

            # acceptance rule: under-target or very high quality
            curr_ratio = self._current_ratio(selected, key=key)
            target_ratio = self.targets.get(key, 0.0)
            if curr_ratio < target_ratio or f.ml_score > high_quality_threshold:
                selected.append(f)
                self.counts[key] += 1
                self.total_selected += 1

        return selected

    def _current_ratio(
        self,
        selected: List[FrameInfo],
        key: Optional[str] = None,
        cover: Optional[str] = None,
    ) -> float:
        if not selected:
            return 0.0
        if cover is not None:
            c = sum(1 for f in selected if f.strata and f.strata[2] == cover)
            return c / len(selected)
        if key is not None:
            c = sum(
                1 for f in selected if f.strata and self.combo_key(*f.strata) == key
            )
            return c / len(selected)
        return 0.0


# -----------------------------------------------------------------------------
# Quality-aware deduplikace (greedy) + cosine threshold
# -----------------------------------------------------------------------------
def auto_eps_from_adjacent_sims(
    frames: List[FrameInfo],
    quantile: Optional[float] = None,
    config: Optional[Dict] = None,
) -> float:
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
    if cosine_threshold is None:
        deduplication_config = (config or {}).get("deduplication", {})
        cosine_threshold = deduplication_config.get("cosine_threshold", 0.85)
    if not frames:
        return []

    # Simple greedy deduplication with fixed cosine threshold
    selected: List[FrameInfo] = []
    for f in sorted(frames, key=lambda x: x.ml_score, reverse=True):
        # Check if similar to any already selected frame
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
    """Deduplicate using DBSCAN on frame embeddings.

    For each DBSCAN cluster we pick the frame with highest ml_score.
    Accepts `eps` in cosine-distance space (i.e. 0..2 for cosine metric as used
    via sklearn metric 'cosine' where distance = 1 - cosine_similarity). If
    eps is None we fall back to `auto_eps_from_adjacent_sims` computed on
    frames sorted by time (approx adjacency heuristic).
    """
    if not frames:
        return []

    # prepare eps
    dedup_conf = (config or {}).get("deduplication", {})
    if eps is None:
        try:
            eps = dedup_conf.get("eps", None)
            if eps is None:
                eps = auto_eps_from_adjacent_sims(frames, config=config)
        except Exception:
            eps = 0.1

    # build embedding matrix
    X = np.vstack([f.embed for f in frames]).astype(np.float32)

    # use sklearn DBSCAN with cosine metric (returns distance = 1 - cosine)
    db = DBSCAN(eps=eps, min_samples=int(min_samples), metric="cosine")
    labels = db.fit_predict(X)

    clusters: Dict[int, List[Tuple[FrameInfo, float]]] = {}
    for f, lab in zip(frames, labels):
        clusters.setdefault(int(lab), []).append((f, f.ml_score))

    selected: List[FrameInfo] = []
    # for noise points label == -1 -> treat each as singleton cluster
    for lab, items in clusters.items():
        # pick highest-ml_score item in cluster
        best = max(items, key=lambda t: t[1])[0]
        selected.append(best)

    # preserve ordering by ml_score descending
    selected = sorted(selected, key=lambda x: x.ml_score, reverse=True)
    return selected


# -----------------------------------------------------------------------------
# End-to-end kurace
# -----------------------------------------------------------------------------
def curate_video(
    video_path: str,
    out_dir: str,
    stride: int,
    target_size: Optional[int],
    min_sharpness: float,
    min_contrast: float,
    novelty_threshold: float,
    manifest_name: str = "manifest.json",
    config: Optional[Dict] = None,
    dedup_method: Optional[str] = None,
    run_params: Optional[Dict] = None,
) -> List[FrameInfo]:
    logger.info(f"Starting video curation: {video_path}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(
        f"Parameters: stride={stride}, target_size={target_size}, min_sharpness={min_sharpness}, "
        f"min_contrast={min_contrast}, novelty_threshold={novelty_threshold}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # 1) Pre-filter
    logger.info("Step 1/5: Pre-filtering frames...")
    candidates = prefilter_candidates(
        video_path=video_path,
        stride=stride,
        min_sharpness=min_sharpness,
        min_contrast=min_contrast,
        config=config,
    )
    if not candidates:
        logger.info("Žádní kandidáti nesplnili pre-filter.")
        return []

    # 2) ML skórování
    logger.info("Step 2/5: Computing ML scores...")
    scorer = MLFrameScorer(novelty_threshold=novelty_threshold, config=config)
    scorer.score(candidates, config=config)

    ml_scores = [f.ml_score for f in candidates]
    logger.debug(
        f"ML scores: min={min(ml_scores):.3f}, max={max(ml_scores):.3f}, mean={np.mean(ml_scores):.3f}"
    )

    # 3) Binning agro os (potřebujeme kvantily pro altitude)
    logger.info("Step 3/5: Computing agro stratification...")
    hf_vals = np.array([f.hf_energy for f in candidates], dtype=np.float32)
    q50 = float(np.quantile(hf_vals, 0.5))
    logger.debug(f"Altitude quantile: q50={q50:.2f}")

    strata_counts = defaultdict(int)
    stratification_config = (config or {}).get("stratification", {})
    thresholds = stratification_config.get("thresholds", {})
    view_threshold = thresholds.get("view_entropy", 1.8)
    cover_threshold = thresholds.get("cover_ratio", 0.5)
    lighting_threshold = thresholds.get("lighting_mean", 115)

    for f in candidates:
        a = bin_altitude(f.hf_energy, q50)
        v = bin_view(f.view_entropy_val, t=view_threshold)
        c = bin_cover(f.green_cover, threshold=cover_threshold)
        l = bin_lighting(f.lighting_mean, threshold=lighting_threshold)
        f.strata = (a, v, c, l)
        strata_counts[f.strata] += 1

    logger.debug(f"Strata distribution: {dict(strata_counts)}")

    # 4) Výběr podle target_size nebo threshold
    cand_sorted = sorted(candidates, key=lambda x: x.ml_score, reverse=True)
    if target_size is None:
        # Threshold-based selection: top % of candidates
        selection_config = (config or {}).get("selection", {})
        threshold_ratio = selection_config.get("threshold_selection_ratio", 0.25)
        threshold_count = max(1, int(len(candidates) * threshold_ratio))
        logger.info(
            f"Step 4/5: Applying threshold-based selection (top {threshold_count} frames, {threshold_ratio:.1%} of candidates)..."
        )
        prelim = cand_sorted[:threshold_count]
        logger.debug(f"Threshold selection: {len(prelim)} frames selected")
    else:
        # Stratified selection
        logger.info("Step 4/5: Applying stratified selection...")
        strat = AgroStratifier(config or {})
        selection_config = (config or {}).get("selection", {})
        stratified_split_ratio = selection_config.get("stratified_split_ratio", 0.5)
        first_size = max(1, int(round(target_size * stratified_split_ratio)))
        first_batch = strat.select(cand_sorted, first_size, config=config)
        logger.debug(
            f"Stratified selection: {len(first_batch)} frames selected from first batch (first_size={first_size})"
        )

        # 5) Doplnění top kvality
        logger.info("Step 5/5: Adding top quality frames...")
        remaining = [f for f in cand_sorted if f not in first_batch]
        top_quality = remaining[: max(1, target_size - len(first_batch))]
        prelim = first_batch + top_quality
        logger.debug(
            f"Added {len(top_quality)} top quality frames, preliminary total: {len(prelim)}"
        )

    # 4) Dedup
    logger.info("Step 4/5: Deduplicating frames...")
    deduplication_config = (config or {}).get("deduplication", {})
    cosine_threshold = deduplication_config.get("cosine_threshold", 0.85)
    # resolve method: CLI-provided dedup_method takes precedence; otherwise use config, default to 'greedy'
    method = (dedup_method or deduplication_config.get("method") or "greedy").lower()
    if method == "dbscan":
        eps = deduplication_config.get("eps", None)
        min_samples = int(deduplication_config.get("min_samples", 1))
        logger.debug(f"Using DBSCAN deduplication eps={eps} min_samples={min_samples}")
        final = deduplicate_dbscan(
            prelim, eps=eps, min_samples=min_samples, config=config
        )
    else:
        logger.debug(
            f"Using greedy deduplication with cosine threshold {cosine_threshold}"
        )
        final = deduplicate_quality_first(prelim, config=config)
    # sort by score
    final = sorted(final, key=lambda x: x.ml_score, reverse=True)

    # If deduplication removed too many frames, refill from remaining candidates (only if target_size specified)
    if target_size is not None and len(final) < target_size:
        logger.debug(
            f"Deduplication reduced count ({len(final)}) below target ({target_size}), attempting refill"
        )
        # consider candidates that were not in prelim or were removed by dedup
        remaining_pool = [f for f in cand_sorted if f not in final]
        # preserve order by ml_score
        remaining_pool = sorted(remaining_pool, key=lambda x: x.ml_score, reverse=True)
        need = target_size - len(final)
        to_add = remaining_pool[:need]
        if to_add:
            final.extend(to_add)
            logger.debug(f"Refilled {len(to_add)} frames to reach target_size")

    # ensure final length does not exceed target (only if target_size specified)
    if target_size is not None:
        final = final[:target_size]
    logger.debug(f"Deduplication: {len(prelim)} → {len(final)} frames")

    # 5) Uložení
    logger.info("Step 5/5: Saving frames and manifest...")
    output_config = (config or {}).get("output", {})
    jpeg_quality = output_config.get("jpeg_quality", 95)
    log_interval = output_config.get("log_intervals", {}).get("frames_saved", 100)

    paths: List[str] = []
    for i, f in enumerate(final):
        fname = f"frame_{i:06d}_src{f.idx:06d}_t{f.t_sec:010.3f}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, f.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        paths.append(path)
        if (i + 1) % log_interval == 0:
            logger.debug(f"Saved {i + 1}/{len(final)} frames")

    manifest = {
        "video": os.path.abspath(video_path),
        "count_candidates": len(candidates),
        "run_params": run_params or {},
        "count_selected": len(final),
        "out_dir": os.path.abspath(out_dir),
        "axes": ["altitude", "view", "cover", "lighting"],
        "frames": [
            {
                "saved_path": os.path.abspath(p),
                "source_index": f.idx,
                "t_sec": f.t_sec,
                "ml_score": f.ml_score,
                "subscores": f.subscores,
                "strata": f.strata,
                "sharpness": f.sharpness,
                "contrast": f.contrast,
                "exposure_score": f.exposure_score,
                "noise_score": f.noise_score,
                "hf_energy": f.hf_energy,
                "view_entropy": f.view_entropy_val,
                "green_cover": f.green_cover,
                "lighting_mean": f.lighting_mean,
            }
            for p, f in zip(paths, final)
        ],
    }
    with open(os.path.join(out_dir, manifest_name), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # compute manifest statistics and log summary
    try:
        ml_scores = (
            np.array([fr["ml_score"] for fr in manifest["frames"]], dtype=np.float32)
            if manifest.get("frames")
            else np.array([], dtype=np.float32)
        )
        mean_ml = float(np.mean(ml_scores)) if ml_scores.size else 0.0
        median_ml = float(np.median(ml_scores)) if ml_scores.size else 0.0
        std_ml = float(np.std(ml_scores)) if ml_scores.size else 0.0

        per_strata: Dict[str, int] = {}
        axis_counts: Dict[str, Dict[str, int]] = {
            "altitude": {},
            "view": {},
            "cover": {},
            "lighting": {},
        }
        for fr in manifest.get("frames", []):
            strata = fr.get("strata") or ()
            # strata may be list after JSON serialization
            if isinstance(strata, (list, tuple)) and len(strata) == 4:
                a, v, c, l = strata
                key = f"altitude:{a}|view:{v}|cover:{c}|lighting:{l}"
                per_strata[key] = per_strata.get(key, 0) + 1
                axis_counts["altitude"][a] = axis_counts["altitude"].get(a, 0) + 1
                axis_counts["view"][v] = axis_counts["view"].get(v, 0) + 1
                axis_counts["cover"][c] = axis_counts["cover"].get(c, 0) + 1
                axis_counts["lighting"][l] = axis_counts["lighting"].get(l, 0) + 1
            else:
                key = "|".join(strata) if strata else "unknown"
                per_strata[key] = per_strata.get(key, 0) + 1

        logger.info(
            "Hotovo. Kandidátů: %d → vybráno: %d. Výstup: %s",
            len(candidates),
            len(final),
            out_dir,
        )
        logger.info(
            "Manifest: %s (frames=%d) ML score mean=%.3f median=%.3f std=%.3f",
            os.path.join(out_dir, manifest_name),
            len(manifest.get("frames", [])),
            mean_ml,
            median_ml,
            std_ml,
        )
        logger.debug("Per-axis distributions: %s", axis_counts)
        # log top strata buckets in debug to avoid huge logs
        top_strata = dict(sorted(per_strata.items(), key=lambda x: -x[1])[:20])
        logger.debug("Per-strata counts (top %d): %s", len(top_strata), top_strata)
    except Exception as e:
        logger.exception("Error computing manifest statistics: %s", e)

    return final


def print_human_readable_statistics(
    manifest_path: str,
    elapsed: float = None,
    params: Optional[argparse.Namespace] = None,
) -> None:
    """Print comprehensive human-readable statistics from manifest.json"""
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load manifest {manifest_path}: {e}")
        return

    print("\n" + "=" * 80)
    print("📊 FRAME CURATION STATISTICS")
    print("=" * 80)

    # Basic counts
    total_candidates = manifest.get("count_candidates", 0)
    total_selected = manifest.get("count_selected", 0)
    frames = manifest.get("frames", [])
    target_size = params.target_size if params else None

    print(f"🎬 Video: {os.path.basename(manifest.get('video', 'Unknown'))}")
    print(f"📁 Output: {manifest.get('out_dir', 'Unknown')}")
    print(
        f"🔢 Selection: {total_selected:,} frames selected from {total_candidates:,} candidates"
    )
    print(
        f"📈 Selection ratio: {(total_selected/total_candidates*100):.1f}%"
        if total_candidates > 0
        else "Selection ratio: N/A"
    )

    if params:
        print(f"\n⚙️ PARAMETERS (explicit or defaults):")
        # prefer run_params from manifest when available
        run_params = manifest.get("run_params") or {}
        print(f"   Config file:         {params.config or 'None'}")

        def fmt(k):
            r = run_params.get(k, {})
            v = r.get("value") if r else getattr(params, k, None)
            src = (
                r.get("source")
                if r
                else ("cli" if getattr(params, k, None) is not None else "default")
            )
            # Determine config file name if source is config
            config_file = getattr(params, "config", None)
            config_file = config_file if config_file else "config.yaml"
            if src == "cli":
                src_disp = "cli"
            else:
                src_disp = config_file
            # Show actual default value from config.yaml if present
            if v is None:
                # Try to get from config defaults
                defaults = {}
                try:
                    import yaml

                    with open(config_file, "r", encoding="utf-8") as f:
                        conf = yaml.safe_load(f)
                        defaults = conf.get("defaults", {})
                except Exception:
                    defaults = {}
                v = defaults.get(k, None)
                src_disp = config_file
            if k == "target_size":
                disp = v if v is not None else "None (threshold-based)"
            else:
                disp = v
            return f"{disp} (source={src_disp})"

        print(f"   Stride:              {fmt('stride')}")
        print(f"   Target size:         {fmt('target_size')}")
        print(f"   Min sharpness:       {fmt('min_sharpness')}")
        print(f"   Min contrast:        {fmt('min_contrast')}")
        print(f"   Novelty threshold:   {fmt('novelty_threshold')}")
        print(f"   Dedup method:        {fmt('dedup_method')}")

    if target_size is not None and total_candidates < target_size:
        print("\n" + "⚠️" + " WARNING: INSUFFICIENT CANDIDATES " + "⚠️")
        print(
            f"The number of candidates ({total_candidates}) was less than the target size ({target_size})."
        )
        print(
            "To get more candidates, consider relaxing pre-filtering criteria (e.g., lower '--min-sharpness')"
        )
        print(
            "or adjusting the '--novelty-threshold' to include more frames in the initial selection."
        )
        print("=" * 80)

    if not frames:
        print("⚠️ No frames found in manifest.")
        return

    # ML Score statistics
    ml_scores = [fr["ml_score"] for fr in frames if "ml_score" in fr]
    if ml_scores:
        print(f"\n🎯 ML QUALITY SCORES:")
        print(f"   Mean: {np.mean(ml_scores):.3f}")
        print(f"   Median: {np.median(ml_scores):.3f}")
        print(f"   Std Dev: {np.std(ml_scores):.3f}")
        print(f"   Range: {min(ml_scores):.3f} - {max(ml_scores):.3f}")

    # Subscores breakdown
    if frames and frames[0].get("subscores"):
        quality_scores = [
            fr["subscores"]["quality"] for fr in frames if fr.get("subscores")
        ]
        novelty_scores = [
            fr["subscores"]["content_novelty"] for fr in frames if fr.get("subscores")
        ]

        if quality_scores:
            print(f"\n📏 SCORE COMPONENTS:")
            print(
                f"   Quality: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}"
            )
            print(
                f"   Novelty: {np.mean(novelty_scores):.3f} ± {np.std(novelty_scores):.3f}"
            )

    # Stratification analysis
    axis_counts = {
        "altitude": defaultdict(int),
        "view": defaultdict(int),
        "cover": defaultdict(int),
        "lighting": defaultdict(int),
    }

    strata_combinations = defaultdict(int)

    for fr in frames:
        strata = fr.get("strata", [])
        if len(strata) == 4:
            a, v, c, l = strata
            axis_counts["altitude"][a] += 1
            axis_counts["view"][v] += 1
            axis_counts["cover"][c] += 1
            axis_counts["lighting"][l] += 1

            combo = f"{a}|{v}|{c}|{l}"
            strata_combinations[combo] += 1

    print(f"\n🏔️ ALTITUDE DISTRIBUTION:")
    for alt, count in sorted(axis_counts["altitude"].items()):
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"   {alt:12}: {count:4} ({pct:5.1f}%)")

    print(f"\n👁️ VIEW ANGLE DISTRIBUTION:")
    for view, count in sorted(axis_counts["view"].items()):
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"   {view:12}: {count:4} ({pct:5.1f}%)")

    print(f"\n🌱 VEGETATION COVER:")
    for cover, count in sorted(axis_counts["cover"].items()):
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"   {cover:12}: {count:4} ({pct:5.1f}%)")

    print(f"\n💡 LIGHTING CONDITIONS:")
    for light, count in sorted(axis_counts["lighting"].items()):
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"   {light:12}: {count:4} ({pct:5.1f}%)")

    # Top strata combinations
    print(f"\n🎯 TOP STRATA COMBINATIONS:")
    top_combos = sorted(strata_combinations.items(), key=lambda x: -x[1])[:10]
    for combo, count in top_combos:
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"   {combo:40}: {count:3} ({pct:4.1f}%)")

    # Technical metrics summary
    if frames:
        sharpness = [fr.get("sharpness", 0) for fr in frames]
        contrast = [fr.get("contrast", 0) for fr in frames]
        exposure = [fr.get("exposure_score", 0) for fr in frames]
        green_cover = [fr.get("green_cover", 0) for fr in frames]

        print(f"\n🔧 TECHNICAL QUALITY METRICS:")
        print(f"   Sharpness:    {np.mean(sharpness):6.1f} ± {np.std(sharpness):5.1f}")
        print(f"   Contrast:     {np.mean(contrast):6.1f} ± {np.std(contrast):5.1f}")
        print(f"   Exposure:     {np.mean(exposure):6.3f} ± {np.std(exposure):5.3f}")
        print(
            f"   Green Cover:  {np.mean(green_cover):6.3f} ± {np.std(green_cover):5.3f}"
        )

    if elapsed is not None:
        mins, secs = divmod(int(elapsed), 60)
        print(f"\n⏱️ Processing time: {mins}m {secs}s")

    print("=" * 80)


# -----------------------------------------------------------------------------
# YAML načtení
# -----------------------------------------------------------------------------
def load_yaml(path: Optional[str]) -> Dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser(defaults: Dict = {}) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ML-Driven Frame Curation — Agro Stratification (MVP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Vstupní video soubor")
    p.add_argument("-o", "--out", help="Výstupní složka")
    p.add_argument(
        "--config", default=None, help="Cesta k YAML konfiguraci (volitelné)"
    )
    p.add_argument("--stride", type=int, help="Vybírat každý N-tý snímek")
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Cílový počet vybraných snímků (None = threshold-based selection)",
    )
    p.add_argument(
        "--min-sharpness",
        type=float,
        help="Minimální ostrost (Variance of Laplacian)",
    )
    p.add_argument("--min-contrast", type=float, help="Minimální kontrast (std gray)")
    p.add_argument(
        "--novelty-threshold",
        type=float,
        help="Práh pro přidání snímku do prototypů pro výpočet novelty (0..1)",
    )
    p.add_argument(
        "--dedup-method",
        choices=["greedy", "dbscan"],
        help="Metoda deduplikace: 'greedy' (výchozí) nebo 'dbscan'",
    )
    p.add_argument("--manifest", help="Název manifest JSON")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.set_defaults(**defaults)
    return p


def main():
    # Load config first to get defaults
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", default=None)
    temp_args, _ = temp_parser.parse_known_args()

    conf = load_yaml(temp_args.config)
    defaults = conf.get("defaults", {})

    ap = build_argparser(defaults=defaults)
    args = ap.parse_args()

    # Set default output folder if not specified
    if args.out is None:
        video_base = os.path.splitext(os.path.basename(args.video))[0]
        args.out = os.path.join("data", "output", video_base)

    # configure logging
    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    log_path = os.path.join(args.out, "curation.log")
    os.makedirs(args.out, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.setLevel(level)

    # Optionally add console handler if --debug is set
    if args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(console_handler)

    logger.info("Starting ML-driven frame curation with agro stratification")
    if args.debug:
        logger.info("Debug mode enabled - detailed logging will be shown")

    if args.config:
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        logger.info("Using default configuration")

    # Collect run parameters with source (cli vs config/default)
    import sys

    defaults_from_conf = defaults or {}
    run_params: Dict[str, Dict[str, object]] = {}
    # list of parameters to report
    keys = [
        "out",
        "stride",
        "target_size",
        "min_sharpness",
        "min_contrast",
        "novelty_threshold",
        "dedup_method",
        "manifest",
    ]
    # Build a mapping from dest to option strings
    dest_to_opts = {}
    for action in ap._actions:
        if action.option_strings:
            for opt in action.option_strings:
                dest_to_opts.setdefault(action.dest, []).append(opt)
    # Build a set of CLI-specified dests
    cli_dests = set()
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("-"):
            for dest, opts in dest_to_opts.items():
                if arg in opts:
                    cli_dests.add(dest)
    for k in keys:
        dest = k
        val = getattr(
            args, dest if hasattr(args, dest) else dest.replace("-", "_"), None
        )
        # normalize explicit null from YAML to Python None
        if val == "null":
            val = None
        if val is None and k in defaults_from_conf:
            val = defaults_from_conf.get(k)
        # Determine source
        if dest in cli_dests:
            source = "cli"
        elif k in defaults_from_conf:
            source = "config"
        else:
            source = "default"
        run_params[k] = {"value": val, "source": source}

    logger.info("Run parameters: %s", {k: v["value"] for k, v in run_params.items()})

    # Ensure all parameters have valid values (fallback if None)
    stride = args.stride if args.stride is not None else 1
    min_sharpness = args.min_sharpness if args.min_sharpness is not None else 80.0
    min_contrast = args.min_contrast if args.min_contrast is not None else 20.0
    novelty_threshold = (
        args.novelty_threshold if args.novelty_threshold is not None else 0.3
    )
    manifest_name = args.manifest if args.manifest is not None else "manifest.json"

    # Run the curation with timing
    start_time = time.time()
    final_frames = curate_video(
        video_path=args.video,
        out_dir=args.out,
        stride=int(stride),
        target_size=args.target_size,
        min_sharpness=float(min_sharpness),
        min_contrast=float(min_contrast),
        novelty_threshold=float(novelty_threshold),
        manifest_name=str(manifest_name),
        config=conf,
        dedup_method=(args.dedup_method if hasattr(args, "dedup_method") else None),
        run_params=run_params,
    )
    elapsed = time.time() - start_time

    # Print human-readable statistics
    manifest_path = os.path.join(args.out, manifest_name)
    if os.path.exists(manifest_path):
        print_human_readable_statistics(manifest_path, elapsed=elapsed, params=args)
    else:
        logger.warning(f"Manifest file not found: {manifest_path}")

    logger.info(f"Frame curation completed successfully! Results saved to: {args.out}")


if __name__ == "__main__":
    main()
