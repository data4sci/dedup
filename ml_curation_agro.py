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
  2) ML skóre (quality + content_novelty + geom_proxy).
  3) Stratifikace dle 4 os (altitude, view, cover, lighting) s cílovými podíly z YAML.
  4) Doplnění top kvality.
  5) Quality-aware deduplikace (DBSCAN s cosine; eps odhad z distribuce).
  6) Uložení snímků + manifest.

Závislosti:
    pip install opencv-python numpy scikit-learn pyyaml

Použití:
    python ml_curation_agro.py input.mp4 -o out_dir --config curation_config.agro.yaml
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

try:
    from sklearn.cluster import DBSCAN

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


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


def exposure_metrics(gray: np.ndarray) -> Tuple[float, float, float]:
    mean = float(np.mean(gray))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)
    under = float(hist[:10].sum())
    over = float(hist[246:].sum())
    return mean, under, over


def exposure_score_from_metrics(
    mean: float, under_frac: float, over_frac: float
) -> float:
    center_penalty = abs(mean - 128.0) / 128.0
    clip_penalty = 2.0 * (under_frac + over_frac)
    raw = 1.0 - min(1.0, 0.6 * center_penalty + 0.4 * clip_penalty)
    return float(max(0.0, min(1.0, raw)))


def estimate_noise_score(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    resid_std = float(np.std(resid))
    score = 1.0 - min(1.0, resid_std / 25.0)
    return float(max(0.0, score))


# -----------------------------------------------------------------------------
# Rychlé embeddingy
# -----------------------------------------------------------------------------
def hsv_histogram(
    bgr: np.ndarray, bins: Tuple[int, int, int] = (16, 16, 16)
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = bins
    h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    hist = np.concatenate([h.ravel(), s.ravel(), v.ravel()]).astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-9
    return hist


def lowres_embedding(bgr: np.ndarray, size: Tuple[int, int] = (64, 36)) -> np.ndarray:
    small = cv2.resize(bgr, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= gray.mean()
    vec = gray.reshape(-1)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.astype(np.float32)


def combined_embed(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = hsv_histogram(bgr)
    low = lowres_embedding(bgr)
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
def altitude_proxy(gray: np.ndarray) -> float:
    # HF energie jako průměr absolutní hodnoty high-pass
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0)
    return float(np.mean(np.abs(hp)))


def view_entropy(gray: np.ndarray, bins: int = 8) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    hist, _ = np.histogram(
        ang.ravel(), bins=bins, range=(0, 2 * np.pi), weights=mag.ravel()
    )
    p = hist / (hist.sum() + 1e-9)
    ent = -np.sum(p * np.log(p + 1e-9))
    return float(ent)


def green_cover_ratio(bgr: np.ndarray) -> float:
    # Excess Green proxy (0..1)
    b, g, r = cv2.split(bgr.astype(np.float32) + 1e-6)
    exg = 2 * g - r - b
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-9)
    thr = 0.6  # lze parametrizovat v YAML
    return float(np.mean(exg_norm >= thr))


def classify_lighting(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def bin_altitude(hf: float, q33: float, q66: float) -> str:
    if hf >= q66:
        return "low"  # hodně detailů -> nízko
    if hf >= q33:
        return "mid"
    return "high"  # málo detailů -> vysoko


def bin_view(ent: float, t1: float = 1.6, t2: float = 1.9) -> str:
    if ent >= t2:
        return "nadir"
    if ent >= t1:
        return "oblique_low"
    return "oblique_high"


def bin_cover(ratio: float) -> str:
    if ratio >= 0.6:
        return "crop_dense"
    if ratio >= 0.25:
        return "crop_sparse"
    return "bare_soil"


def bin_lighting(mean_int: float) -> str:
    if mean_int < 70:
        return "dark"
    if mean_int > 160:
        return "bright"
    return "normal"


# -----------------------------------------------------------------------------
# Streamování videa + prefilter + výpočet proxy
# -----------------------------------------------------------------------------
def iter_video_frames(
    video_path: str, stride: int = 1
) -> Iterable[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
) -> List[FrameInfo]:
    logger.debug(
        f"Starting pre-filter with stride={stride}, min_sharpness={min_sharpness}, min_contrast={min_contrast}"
    )
    out: List[FrameInfo] = []
    total_frames = 0
    filtered_out = 0

    for idx, t_sec, bgr in iter_video_frames(video_path, stride=stride):
        total_frames += 1
        if total_frames % 1000 == 0:
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

        mean, under, over = exposure_metrics(gray)
        expo = exposure_score_from_metrics(mean, under, over)
        noise = estimate_noise_score(gray)
        hsv, low, emb = combined_embed(bgr)

        # agro proxies (continuous)
        hf = altitude_proxy(gray)
        ent = view_entropy(gray)
        gcr = green_cover_ratio(bgr)
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
    def __init__(self, novelty_memory: int = 64, novelty_threshold: float = 0.3):
        self.prototypes: List[np.ndarray] = []
        self.novelty_memory = int(novelty_memory)
        self.novelty_threshold = novelty_threshold

    @staticmethod
    def _scale(x: float, lo: float, hi: float) -> float:
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo + 1e-9))))

    def _quality_score(self, f: FrameInfo) -> float:
        s = self._scale(f.sharpness, 80.0, 300.0)
        c = self._scale(f.contrast, 20.0, 80.0)
        e = f.exposure_score
        n = f.noise_score
        return 0.35 * s + 0.30 * c + 0.25 * e + 0.10 * n

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

    def score(self, frames: List[FrameInfo]) -> None:
        logger.debug(f"Starting ML scoring for {len(frames)} frames")
        for i, f in enumerate(frames):
            if i % 500 == 0 and i > 0:
                logger.debug(f"ML scoring progress: {i}/{len(frames)} frames")

            q = self._quality_score(f)
            nov = self._novelty_score(f.embed)
            geom = self._geom_score(f)
            total = 0.4 * q + 0.45 * nov + 0.15 * geom
            f.ml_score = float(total)
            f.subscores = {"quality": q, "content_novelty": nov, "geom": geom}

            logger.debug(
                f"Frame {f.idx}: quality={q:.3f}, novelty={nov:.3f}, geom={geom:.3f} → ML_score={total:.3f}"
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
    YAML formát (viz curation_config.agro.yaml):
      stratification:
        axes:
          altitude: [low, mid, high]
          view: [nadir, oblique_low, oblique_high]
          cover: [bare_soil, crop_sparse, crop_dense]
          lighting: [dark, normal, bright]
        targets:
          "altitude:low|view:nadir|cover:crop_dense|lighting:normal": 0.24
          "*": 0.76
        limits:
          windy_max_ratio: 0.15
          bare_soil_max_ratio: 0.20
    """

    def __init__(self, config: Dict):
        sconf = (config or {}).get("stratification", {})
        self.axes = sconf.get(
            "axes",
            {
                "altitude": ["low", "mid", "high"],
                "view": ["nadir", "oblique_low", "oblique_high"],
                "cover": ["bare_soil", "crop_sparse", "crop_dense"],
                "lighting": ["dark", "normal", "bright"],
            },
        )
        self.targets_raw: Dict[str, float] = sconf.get("targets", {"*": 1.0})
        self.limits = sconf.get(
            "limits", {"windy_max_ratio": 0.15, "bare_soil_max_ratio": 0.20}
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
        self, frames_sorted: List[FrameInfo], target_size: int
    ) -> List[FrameInfo]:
        selected: List[FrameInfo] = []
        for f in frames_sorted:
            if len(selected) >= target_size:
                break

            # derive key and ratios
            a, v, c, l = f.strata  # type: ignore
            key = self.combo_key(a, v, c, l)

            # enforce limits (example: bare_soil cap)
            if c == "bare_soil":
                if self._current_ratio(selected, cover="bare_soil") >= self.limits.get(
                    "bare_soil_max_ratio", 1.0
                ):
                    continue

            # acceptance rule: under-target or very high quality
            curr_ratio = self._current_ratio(selected, key=key)
            target_ratio = self.targets.get(key, 0.0)
            if curr_ratio < target_ratio or f.ml_score > 0.95:
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
# Quality-aware deduplikace (DBSCAN) + eps odhad
# -----------------------------------------------------------------------------
def auto_eps_from_adjacent_sims(
    frames: List[FrameInfo], quantile: float = 0.90
) -> float:
    if len(frames) < 3:
        return 0.10
    sims = []
    for i in range(len(frames) - 1):
        sims.append(cosine_similarity(frames[i].embed, frames[i + 1].embed))
    Q = float(np.quantile(np.array(sims, dtype=np.float32), quantile))
    eps = max(0.02, min(0.30, 1.0 - Q))
    return eps


def deduplicate_quality_first(
    frames: List[FrameInfo], keep_frac_per_cluster: float = 0.10
) -> List[FrameInfo]:
    if not frames:
        return []
    if not HAVE_SKLEARN:
        logger.warning("scikit-learn není k dispozici; greedy fallback dedup.")
        selected: List[FrameInfo] = []
        pivot: Optional[FrameInfo] = None
        thr = 1.0 - auto_eps_from_adjacent_sims(frames, 0.90)
        for f in frames:
            if pivot is None:
                selected.append(f)
                pivot = f
                continue
            if cosine_similarity(f.embed, pivot.embed) < thr:
                selected.append(f)
                pivot = f
        return selected

    X = np.stack([f.embed for f in frames], axis=0)
    eps = auto_eps_from_adjacent_sims(frames, 0.90)
    clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(X)
    labels = clustering.labels_

    clusters: Dict[int, List[FrameInfo]] = defaultdict(list)
    for f, lab in zip(frames, labels):
        clusters[lab].append(f)

    selected: List[FrameInfo] = []
    for lab, items in clusters.items():
        if lab == -1:
            selected.extend(items)  # singletons
            continue
        items_sorted = sorted(items, key=lambda x: x.ml_score, reverse=True)
        k = max(1, int(round(len(items_sorted) * keep_frac_per_cluster)))
        selected.extend(items_sorted[:k])
    return selected


# -----------------------------------------------------------------------------
# End-to-end kurace
# -----------------------------------------------------------------------------
def curate_video(
    video_path: str,
    out_dir: str,
    stride: int,
    target_size: int,
    min_sharpness: float,
    min_contrast: float,
    novelty_threshold: float,
    manifest_name: str = "manifest.json",
    config: Optional[Dict] = None,
) -> List[FrameInfo]:
    logger.info(f"Starting video curation: {video_path}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(
        f"Parameters: stride={stride}, target_size={target_size}, min_sharpness={min_sharpness}, "
        f"min_contrast={min_contrast}, novelty_threshold={novelty_threshold}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # 1) Pre-filter
    logger.info("Step 1/7: Pre-filtering frames...")
    candidates = prefilter_candidates(
        video_path=video_path,
        stride=stride,
        min_sharpness=min_sharpness,
        min_contrast=min_contrast,
    )
    if not candidates:
        logger.info("Žádní kandidáti nesplnili pre-filter.")
        return []

    # 2) ML skórování
    logger.info("Step 2/7: Computing ML scores...")
    scorer = MLFrameScorer(novelty_memory=64, novelty_threshold=novelty_threshold)
    scorer.score(candidates)

    ml_scores = [f.ml_score for f in candidates]
    logger.debug(
        f"ML scores: min={min(ml_scores):.3f}, max={max(ml_scores):.3f}, mean={np.mean(ml_scores):.3f}"
    )

    # 3) Binning agro os (potřebujeme kvantily pro altitude)
    logger.info("Step 3/7: Computing agro stratification...")
    hf_vals = np.array([f.hf_energy for f in candidates], dtype=np.float32)
    q33, q66 = np.quantile(hf_vals, [0.33, 0.66]).tolist()
    logger.debug(f"Altitude quantiles: q33={q33:.2f}, q66={q66:.2f}")

    strata_counts = defaultdict(int)
    for f in candidates:
        a = bin_altitude(f.hf_energy, q33, q66)
        v = bin_view(f.view_entropy_val, t1=1.6, t2=1.9)
        c = bin_cover(f.green_cover)
        l = bin_lighting(f.lighting_mean)
        f.strata = (a, v, c, l)
        strata_counts[f.strata] += 1

    logger.debug(f"Strata distribution: {dict(strata_counts)}")

    # 4) Stratifikace z YAML
    logger.info("Step 4/7: Applying stratified selection...")
    strat = AgroStratifier(config or {})
    cand_sorted = sorted(candidates, key=lambda x: x.ml_score, reverse=True)
    first_batch = strat.select(cand_sorted, target_size=max(1, target_size // 2))
    logger.debug(
        f"Stratified selection: {len(first_batch)} frames selected from first batch"
    )

    # 5) Doplnění top kvality
    logger.info("Step 5/7: Adding top quality frames...")
    remaining = [f for f in cand_sorted if f not in first_batch]
    top_quality = remaining[: max(1, target_size - len(first_batch))]
    prelim = first_batch + top_quality
    logger.debug(
        f"Added {len(top_quality)} top quality frames, preliminary total: {len(prelim)}"
    )

    # 6) Dedup
    logger.info("Step 6/7: Deduplicating frames...")
    eps_used = auto_eps_from_adjacent_sims(prelim, 0.90) if prelim else 0.1
    logger.debug(f"Using DBSCAN eps={eps_used:.3f} for deduplication")
    final = deduplicate_quality_first(prelim, keep_frac_per_cluster=0.10)
    # sort by score
    final = sorted(final, key=lambda x: x.ml_score, reverse=True)

    # If deduplication removed too many frames, refill from remaining candidates
    if len(final) < target_size:
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

    # ensure final length does not exceed target
    final = final[:target_size]
    logger.debug(f"Deduplication: {len(prelim)} → {len(final)} frames")

    # 7) Uložení
    logger.info("Step 7/7: Saving frames and manifest...")
    paths: List[str] = []
    for i, f in enumerate(final):
        fname = f"frame_{i:06d}_src{f.idx:06d}_t{f.t_sec:010.3f}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, f.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(path)
        if (i + 1) % 100 == 0:
            logger.debug(f"Saved {i + 1}/{len(final)} frames")

    manifest = {
        "video": os.path.abspath(video_path),
        "count_candidates": len(candidates),
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
    target_size = int(params.target_size) if params else None

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
        print(f"\n⚙️ PARAMETERS USED:")
        print(f"   Config file:         {params.config or 'None'}")
        print(f"   Stride:              {params.stride}")
        print(f"   Target size:         {params.target_size}")
        print(f"   Min sharpness:       {params.min_sharpness}")
        print(f"   Min contrast:        {params.min_contrast}")
        print(f"   Novelty threshold:   {params.novelty_threshold}")

    if target_size and total_candidates < target_size:
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
        geom_scores = [fr["subscores"]["geom"] for fr in frames if fr.get("subscores")]

        if quality_scores:
            print(f"\n📏 SCORE COMPONENTS:")
            print(
                f"   Quality: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}"
            )
            print(
                f"   Novelty: {np.mean(novelty_scores):.3f} ± {np.std(novelty_scores):.3f}"
            )
            print(
                f"   Geometry: {np.mean(geom_scores):.3f} ± {np.std(geom_scores):.3f}"
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
    p.add_argument("--target-size", type=int, help="Cílový počet vybraných snímků")
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

    # Run the curation with timing
    start_time = time.time()
    final_frames = curate_video(
        video_path=args.video,
        out_dir=args.out,
        stride=int(args.stride),
        target_size=int(args.target_size),
        min_sharpness=float(args.min_sharpness),
        min_contrast=float(args.min_contrast),
        novelty_threshold=float(args.novelty_threshold),
        manifest_name=str(args.manifest),
        config=conf,
    )
    elapsed = time.time() - start_time

    # Print human-readable statistics
    manifest_path = os.path.join(args.out, args.manifest)
    if os.path.exists(manifest_path):
        print_human_readable_statistics(manifest_path, elapsed=elapsed, params=args)
    else:
        logger.warning(f"Manifest file not found: {manifest_path}")

    logger.info(f"Frame curation completed successfully! Results saved to: {args.out}")


if __name__ == "__main__":
    main()
