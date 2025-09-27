#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced Frame Extractor — Agro ML Dataset Curation (orchestrator)
================================================================

Cílem tohoto modulu je orchestrace procesu kurace snímků z droních videí
pro vytvoření vyváženého datasetu pro ML trénink. Implementované kroky
odpovídají popisu v README.md a používají jednotnou terminologii:

- Pre-filter (prefilter): odstranění snímků s nízkou kvalitou (ostrost, kontrast).
- Quality metrics: výpočet metrik kvality (exposure, noise, lighting mean).
- Embeddings: výpočet embeddingů (HSV hist + LowRes vector + případné modelové embed).
- Agro-proxy: proxy metriky použité pro stratifikaci — altitude (hf_energy), view entropy,
  green cover ratio, lighting mean.
- ML skórování: ohodnocení kandidátů pomocí `MLFrameScorer` (kombinace kvality a novosti).
- Selection:
  - Stratified selection (pokud je zadán `target_size`) → používá `AgroStratifier`.
  - Threshold-based selection (pokud `target_size` není zadán) → top % kandidátů.
- Deduplication: odstranění vizuálních duplicit (`greedy` nebo `dbscan`).
- Output / manifest (level-0): uloží vybrané snímky a vygeneruje `manifest.json`
  se strukturou: `input_video`, `run_params`, `task_summary`, `frames`.

Důležité poznámky:
- Terminologie v komentářích a docstringách byla sjednocena podle README.
- Tento soubor mění pouze komentáře a docstringy; chování kódu zůstává beze změny.
"""
from __future__ import annotations

import os
import sys
import shutil
import json
import argparse
import logging
import time
from collections import defaultdict
from typing import List, Dict, Optional

import cv2
import numpy as np
import yaml

# Importy z refaktorované knihovny `bfe` — nízkoúrovňové implementace pro jednotlivé kroky.
from bfe.frame_info import FrameInfo
from bfe.video_io import iter_video_frames
from bfe.quality_metrics import (
    variance_of_laplacian,
    estimate_contrast,
    exposure_metrics,
    exposure_score_from_metrics,
    estimate_noise_score,
)
from bfe.embeddings import combined_embed
from bfe.proxies import altitude_proxy, view_entropy, green_cover_ratio
from bfe.scoring import MLFrameScorer
from bfe.binning import bin_altitude, bin_view, bin_cover, bin_lighting
from bfe.stratification import AgroStratifier
from bfe.deduplication import deduplicate_quality_first, deduplicate_dbscan

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Krok 1: Pre-filter + výpočet metrik a embeddingů (prefilter_and_process_frames)
# -----------------------------------------------------------------------------


def prefilter_and_process_frames(
    video_path: str,
    stride: int = 1,
    min_sharpness: float = 80.0,
    min_contrast: float = 20.0,
    config: Optional[Dict] = None,
) -> List[FrameInfo]:
    """
    Iteruje video a provádí pre-filter + kompletní výpočet metrík, agro-proxy a embeddingů.

    Tento krok provede:
      - Pre-filter (ostrost: variance of Laplacian, kontrast: odhad rozptylu šedi).
      - Pokud snímek projde pre-filterem, spočítá exposure metrics, exposure_score,
        noise_score a lighting_mean.
      - Spočítá embeddingy: HSV histogram, low-resolution vector a případné modelové embed.
      - Spočítá agro-proxy: altitude proxy (hf_energy), view entropy, green cover ratio.

    Args:
        video_path (str): Cesta k vstupnímu videu.
        stride (int): Vybírat každý N-tý snímek (redukuje množství kandidátů).
        min_sharpness (float): Minimální práh pro ostrost (Variance of Laplacian).
        min_contrast (float): Minimální práh pro kontrast (std-dev šedé).
        config (Optional[Dict]): Konfigurační slovník (config.yaml).

    Returns:
        List[FrameInfo]: Seznam kandidátů (FrameInfo) s vyplněnými metrikami, proxy a embeddingy.
    """
    logger.info(f"Starting pre-filter & processing for video: {video_path}")
    logger.info(
        f"Parameters: stride={stride}, min_sharpness={min_sharpness}, min_contrast={min_contrast}"
    )

    candidates: List[FrameInfo] = []
    total_frames = 0
    filtered_out = 0

    # Konfigurační pomocné hodnoty (výstup + proxy nastavení)
    output_config = (config or {}).get("output", {})
    log_intervals = output_config.get("log_intervals", {})
    frames_processed_interval = log_intervals.get("frames_processed", 1000)
    proxies_config = (config or {}).get("proxies", {})
    view_bins = proxies_config.get("view_entropy_bins", 8)
    green_threshold = proxies_config.get("green_cover_threshold", 0.6)

    # Iterace přes snímky z videa (stroming, memory-friendly)
    for idx, t_sec, bgr in iter_video_frames(video_path, stride=stride, config=config):
        total_frames += 1
        if total_frames % frames_processed_interval == 0:
            logger.debug(
                f"Processed {total_frames} frames, kept {len(candidates)} candidates"
            )

        # Základní quality metrics (grayscale pro většinu metrik)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)  # ostrost proxy
        contrast = estimate_contrast(gray)  # kontrast proxy

        # Pre-filter: vyřadíme snímky s nízkou ostrostí nebo kontrastem.
        if sharpness < min_sharpness or contrast < min_contrast:
            filtered_out += 1
            logger.debug(
                f"Frame {idx} filtered out: sharpness={sharpness:.1f} (min={min_sharpness}), "
                f"contrast={contrast:.1f} (min={min_contrast})"
            )
            continue

        # Pokročilé metriky kvality
        mean, under, over = exposure_metrics(gray, config=config)
        exposure_score = exposure_score_from_metrics(mean, under, over, config=config)
        noise_score = estimate_noise_score(gray, config=config)

        # Embeddings: kombinujeme HSV histogram + lowres vector + (volitelně) model embed
        hsv_hist, lowres_vec, embed = combined_embed(bgr, config=config)

        # Agro-proxy metriky používané pro stratifikaci
        hf_energy = altitude_proxy(gray, config=config)  # altitude proxy (hf_energy)
        view_entropy_val = view_entropy(gray, bins=view_bins, config=config)
        green_cover = green_cover_ratio(bgr, threshold=green_threshold, config=config)
        lighting_mean = float(np.mean(gray))

        logger.debug(
            f"Frame {idx}: sharpness={sharpness:.1f}, contrast={contrast:.1f}, exposure_score={exposure_score:.3f}, "
            f"hf_energy={hf_energy:.2f}, view_entropy={view_entropy_val:.2f}, green_cover={green_cover:.3f}"
        )

        # Uložení kandidáta s kompletními metrikami/embeddingy/proxy
        candidates.append(
            FrameInfo(
                idx=idx,
                t_sec=t_sec,
                bgr=bgr,
                gray=gray,
                sharpness=sharpness,
                contrast=contrast,
                exposure_score=exposure_score,
                noise_score=noise_score,
                hsv_hist=hsv_hist,
                lowres_vec=lowres_vec,
                embed=embed,
                hf_energy=hf_energy,
                view_entropy_val=view_entropy_val,
                green_cover=green_cover,
                lighting_mean=lighting_mean,
            )
        )

    logger.info(
        f"Pre-filter completed: from {total_frames} frames kept {len(candidates)} candidates, "
        f"{filtered_out} filtered out."
    )
    return candidates


# -----------------------------------------------------------------------------
# Krok 2: Binning / přiřazení strat (assign_strata_to_frames)
# -----------------------------------------------------------------------------


def assign_strata_to_frames(
    frames: List[FrameInfo], config: Optional[Dict] = None
) -> None:
    """
    Přiřadí každému kandidátnímu snímku jeho stratum (kategorii) na základě agro-proxy.

    Popis:
      - Pro každou osu (altitude, view, cover, lighting) provede binning pomocí
        utilit z `bfe.binning`.
      - `altitude` binning používá kvantil (defaultně medián q=0.5), který je
        počítán z celého datasetu kandidátů — to zajistí relativní rozdělení
        podle scénáře z daného videa.

    Args:
        frames (List[FrameInfo]): Seznam kandidátů s vyplněnými agro-proxy.
        config (Optional[Dict]): Konfigurační slovník (může obsahovat stratification.thresholds).

    Returns:
        None -- funkce mutuje objekty FrameInfo (nastaví `strata` atribut).
    """
    if not frames:
        return

    logger.info("Starting strata computation and assignment...")

    stratification_config = (config or {}).get("stratification", {})
    thresholds = stratification_config.get("thresholds", {})
    view_threshold = thresholds.get("view_entropy", 1.8)
    cover_threshold = thresholds.get("cover_ratio", 0.5)
    lighting_threshold = thresholds.get("lighting_mean", 115)

    # Altitude quantile (medián default). TODO: parametrizovat kvantil v configu, pokud požadováno.
    hf_vals = np.array([f.hf_energy for f in frames], dtype=np.float32)
    altitude_q50 = float(np.quantile(hf_vals, 0.5))
    logger.debug(f"Altitude quantile (q50 of hf_energy): {altitude_q50:.2f}")

    strata_counts = defaultdict(int)
    for f in frames:
        # Binning podle os — výsledkem jsou štítky (např. 'low','mid','high')
        a = bin_altitude(f.hf_energy, altitude_q50)
        v = bin_view(f.view_entropy_val, t=view_threshold)
        c = bin_cover(f.green_cover, threshold=cover_threshold)
        l = bin_lighting(f.lighting_mean, threshold=lighting_threshold)
        f.strata = (a, v, c, l)
        strata_counts[f.strata] += 1

    logger.debug(f"Strata distribution: {dict(strata_counts)}")
    logger.info("Strata assignment completed.")


# -----------------------------------------------------------------------------
# Krok 3: Selection (stratified nebo threshold-based) + Deduplication
# -----------------------------------------------------------------------------


def select_and_deduplicate(
    frames: List[FrameInfo],
    target_size: Optional[int],
    config: Optional[Dict] = None,
    dedup_method: Optional[str] = None,
) -> List[FrameInfo]:
    """
    Provede finální výběr snímků (selection) a následnou deduplikaci.

    Selection:
      - Pokud je `target_size` zadán: stratifikovaný výběr pomocí `AgroStratifier`
        (cílem je dosáhnout rovnoměrné zastoupení napříč stratami).
      - Pokud `target_size` není zadán: threshold-based selection — vybere se
        top N% kandidátů podle ML skóre (nastavitelné přes `selection.threshold_selection_ratio`).

    Deduplication:
      - Metoda `greedy` (quality-first s prahovou kosinovou podobností).
      - Metoda `dbscan` (shlukování embeddingů přes DBSCAN).

    Args:
        frames (List[FrameInfo]): Kandidáti (očekává se, že mají `ml_score` vyplněné).
        target_size (Optional[int]): Cílový počet vybraných snímků (pokud None → threshold-based).
        config (Optional[Dict]): Konfigurační slovník.
        dedup_method (Optional[str]): Přepsat metodu deduplikace ('greedy'|'dbscan').

    Returns:
        List[FrameInfo]: Finalní seznam vybraných (a deduplikovaných) FrameInfo objektů.
    """
    if not frames:
        return []

    # Seřazení kandidátů podle ML skóre (nejlepší první)
    frames_sorted = sorted(frames, key=lambda x: x.ml_score, reverse=True)

    # --- Preliminary selection ---
    if target_size is None:
        # Threshold-based selection: vezmeme top % kandidátů podle konfigurace
        selection_config = (config or {}).get("selection", {})
        threshold_ratio = selection_config.get("threshold_selection_ratio", 0.25)
        threshold_count = max(1, int(len(frames) * threshold_ratio))
        logger.info(
            f"Threshold-based selection: selecting top {threshold_count} frames ({threshold_ratio:.1%} of candidates)."
        )
        preliminary_selection = frames_sorted[:threshold_count]
    else:
        # Stratified selection: AgroStratifier zajistí rozmanitost napříč stratami
        logger.info(f"Stratified selection to reach target size {target_size}.")
        stratifier = AgroStratifier(config or {})
        preliminary_selection = stratifier.select(
            frames_sorted, target_size, config=config
        )
        logger.debug(
            f"Stratified selection produced {len(preliminary_selection)} frames."
        )

        # Pokud stratifikace vrátila méně než požadované, doplníme top-quality kandidáty
        if len(preliminary_selection) < target_size:
            needed = target_size - len(preliminary_selection)
            remaining_candidates = [
                f for f in frames_sorted if f not in preliminary_selection
            ]
            to_add = remaining_candidates[:needed]
            preliminary_selection.extend(to_add)
            logger.debug(f"Added {len(to_add)} top-quality frames to meet target.")

    logger.info(f"Preliminary selection size: {len(preliminary_selection)}")

    # --- Deduplication ---
    logger.info("Starting deduplication...")
    deduplication_config = (config or {}).get("deduplication", {})
    method = (dedup_method or deduplication_config.get("method") or "greedy").lower()

    if method == "dbscan":
        eps = deduplication_config.get("eps", None)
        min_samples = int(deduplication_config.get("min_samples", 1))
        logger.info(
            f"Using DBSCAN deduplication (eps={eps}, min_samples={min_samples})."
        )
        final_selection = deduplicate_dbscan(
            preliminary_selection, eps=eps, min_samples=min_samples, config=config
        )
    else:
        cosine_threshold = deduplication_config.get("cosine_threshold", 0.85)
        logger.info(
            f"Using greedy deduplication (cosine_threshold={cosine_threshold})."
        )
        final_selection = deduplicate_quality_first(
            preliminary_selection, cosine_threshold=cosine_threshold, config=config
        )

    logger.info(f"After deduplication {len(final_selection)} frames remain.")

    # --- Fill up after deduplication if necessary ---
    if target_size is not None and len(final_selection) < target_size:
        needed = target_size - len(final_selection)
        logger.debug(
            f"Final selection ({len(final_selection)}) below target ({target_size}). Attempting to fill {needed} frames."
        )

        remaining_pool = [f for f in frames_sorted if f not in final_selection]
        to_add = remaining_pool[:needed]

        if to_add:
            final_selection.extend(to_add)
            logger.debug(
                f"Added {len(to_add)} frames post-deduplication, final count: {len(final_selection)}."
            )

    # Finální seřazení podle ml_score a oříznutí na target_size (pokud zadáno)
    final_selection = sorted(final_selection, key=lambda x: x.ml_score, reverse=True)
    if target_size is not None:
        final_selection = final_selection[:target_size]

    return final_selection


# -----------------------------------------------------------------------------
# Krok 4: Uložení výsledků + generace manifest.json (level-0 manifest)
# -----------------------------------------------------------------------------


def save_results(
    frames: List[FrameInfo],
    out_dir: str,
    video_path: str,
    num_candidates: int,
    run_params: Dict,
    manifest_name: str = "manifest.json",
    config: Optional[Dict] = None,
) -> None:
    """
    Uloží vybrané snímky do adresáře a vytvoří manifest (level-0).

    Manifest level-0 klíče:
      - input_video: absolutní cesta k původnímu videu
      - run_params: map parametrů -> {"value": ..., "source": ...}
      - task_summary: agregace (candidates_count, selected_count, axes_summary, ml_score apod.)
      - frames: pole objektů s metadaty pro každý uložený snímek

    Args:
        frames (List[FrameInfo]): Finalní seznam vybraných FrameInfo.
        out_dir (str): Výstupní adresář.
        video_path (str): Původní video (use to store in manifest).
        num_candidates (int): Počet kandidátů po pre-filter kroku.
        run_params (Dict): Parametry běhu se zdroji (cli/config/default).
        manifest_name (str): Název souboru manifestu (výchozí "manifest.json").
        config (Optional[Dict]): Konfigurace (používají se např. jpeg_quality).
    """
    if not frames:
        logger.warning("No frames to save.")
        return

    logger.info(f"Saving {len(frames)} selected frames to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    output_config = (config or {}).get("output", {})
    jpeg_quality = output_config.get("jpeg_quality", 95)
    log_interval = output_config.get("log_intervals", {}).get("frames_saved", 100)

    saved_paths = []
    for i, f in enumerate(frames):
        # Jmenný formát souboru zahrnuje pořadové číslo, původní index a čas
        fname = f"frame_{i:06d}_src{f.idx:06d}_t{f.t_sec:010.3f}.jpg"
        path = os.path.join(out_dir, fname)
        # Uložíme JPEG s nastavenou kvalitou
        cv2.imwrite(path, f.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        saved_paths.append(path)
        if (i + 1) % log_interval == 0:
            logger.debug(f"Saved {i + 1}/{len(frames)} frames")

    # Agregace: strata distribution (kombinované klíče) + axes_summary (po osách)
    strata_distribution: Dict[str, int] = {}
    axes_summary: Dict[str, Dict[str, int]] = {
        "altitude": {},
        "view": {},
        "cover": {},
        "lighting": {},
    }

    for f in frames:
        if not f.strata:
            continue
        a, v, c, l = f.strata  # type: ignore
        combo_key = f"altitude:{a}|view:{v}|cover:{c}|lighting:{l}"
        strata_distribution[combo_key] = strata_distribution.get(combo_key, 0) + 1

        axes_summary["altitude"][a] = axes_summary["altitude"].get(a, 0) + 1
        axes_summary["view"][v] = axes_summary["view"].get(v, 0) + 1
        axes_summary["cover"][c] = axes_summary["cover"].get(c, 0) + 1
        axes_summary["lighting"][l] = axes_summary["lighting"].get(l, 0) + 1

    # Task summary: agregace metrik a statistiky ML skóre
    task_summary = {
        "video": os.path.basename(video_path),
        "output_dir": os.path.abspath(out_dir),
        "candidates_count": num_candidates,
        "selected_count": len(frames),
        "selection_ratio": (
            f"{(len(frames) / num_candidates * 100):.1f}%"
            if num_candidates > 0
            else "N/A"
        ),
        "ml_score": {
            "average": (
                f"{np.mean([f.ml_score for f in frames]):.3f}" if frames else "N/A"
            ),
            "median": (
                f"{np.median([f.ml_score for f in frames]):.3f}" if frames else "N/A"
            ),
            "range": (
                f"{min([f.ml_score for f in frames]):.3f} - {max([f.ml_score for f in frames]):.3f}"
                if frames
                else "N/A"
            ),
        },
        "axes_summary": axes_summary,
        "strata_distribution": strata_distribution,
    }

    # Sestavení manifestu (level-0)
    manifest = {
        "input_video": os.path.abspath(video_path),
        "run_params": run_params or {},
        "task_summary": task_summary,
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
            for p, f in zip(saved_paths, frames)
        ],
    }

    # Evaluate stratification targets (if present in config) and attach to manifest.
    try:
        targets_eval = evaluate_targets_from_config(
            strata_distribution=strata_distribution,
            selected_count=len(frames),
            candidates_count=num_candidates,
            config=config,
        )
        manifest["targets_evaluation"] = targets_eval
    except Exception as e:
        logger.debug(f"Targets evaluation skipped due to error: {e}")

    manifest_path = os.path.join(out_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    logger.info(f"Manifest saved to: {manifest_path}")


# -----------------------------------------------------------------------------
# Helper: targets evaluation for stratification.targets
# -----------------------------------------------------------------------------
def _cartesian_strata_combinations(axes: Dict[str, List[str]]) -> List[str]:
    """
    Build combo keys in the same format used in manifest ('altitude:low|view:nadir|cover:dense|lighting:bright').
    The axes dict is expected to have the keys in the order: altitude, view, cover, lighting.
    """
    import itertools

    axis_names = list(axes.keys())
    axis_values = [axes[k] for k in axis_names]
    combos = []
    for vals in itertools.product(*axis_values):
        parts = [f"{name}:{val}" for name, val in zip(axis_names, vals)]
        combos.append("|".join(parts))
    return combos


def evaluate_targets_from_config(
    strata_distribution: Dict[str, int],
    selected_count: int,
    candidates_count: int,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Compute expected counts from per-axis `targets_axes`.
    Only per-axis schema is supported.

    Behavior:
      - Use per-axis weights (missing values -> uniform).
      - Combine per-axis weights by product to get combo ratios, then allocate
        expected counts using largest-remainder to ensure sum(expected)=selected_count.
    """
    if not config:
        return {"success": True, "reason": "no_config"}

    strat_cfg = (config or {}).get("stratification", {})
    axes_cfg = strat_cfg.get("axes", {})
    targets_axes_cfg = strat_cfg.get("targets_axes", {})

    if not axes_cfg:
        return {"success": True, "reason": "no_axes_defined"}

    # Build all possible combos in the same key format
    all_combos = _cartesian_strata_combinations(axes_cfg)

    # Helper: normalize axis dict
    def _normalize(d: Dict[str, float], axis_vals: List[str]) -> Dict[str, float]:
        for v in axis_vals:
            d.setdefault(v, 1.0)
        s = sum(float(x) for x in d.values()) + 1e-12
        if s == 0:
            n = len(axis_vals) or 1
            return {v: 1.0 / n for v in axis_vals}
        return {v: float(d.get(v, 0.0)) / s for v in axis_vals}

    # 1) Derive per-axis normalized weights
    per_axis_weights: Dict[str, Dict[str, float]] = {}
    for axis, vals in axes_cfg.items():
        requested = targets_axes_cfg.get(axis, {})
        per_axis_weights[axis] = _normalize(
            {k: float(v) for k, v in requested.items()}, vals
        )

    # ensure all axes present and normalized
    for axis, vals in axes_cfg.items():
        if axis not in per_axis_weights:
            per_axis_weights[axis] = {v: 1.0 / max(1, len(vals)) for v in vals}
        else:
            per_axis_weights[axis] = _normalize(per_axis_weights[axis], vals)

    # 2) compute combo ratios by product of per-axis weights
    combo_ratios: Dict[str, float] = {}
    for combo in all_combos:
        prod = 1.0
        for part in combo.split("|"):
            k, v = part.split(":", 1)
            prod *= per_axis_weights.get(k, {}).get(v, 0.0)
        combo_ratios[combo] = prod
    # normalize
    total = sum(combo_ratios.values()) + 1e-12
    if total == 0:
        n = len(combo_ratios) or 1
        combo_ratios = {c: 1.0 / n for c in combo_ratios}
    else:
        combo_ratios = {c: float(r) / total for c, r in combo_ratios.items()}

    # 3) allocate expected counts using Largest Remainder method to match selected_count
    raw_expected = {c: combo_ratios[c] * selected_count for c in combo_ratios}
    floored = {c: int(float(raw_expected[c])) for c in raw_expected}
    remainders = {c: raw_expected[c] - floored[c] for c in raw_expected}
    allocated = sum(floored.values())
    to_allocate = max(0, selected_count - allocated)
    for c, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True)[
        :to_allocate
    ]:
        floored[c] += 1

    expected_counts = {c: int(floored[c]) for c in floored}
    actual_counts = {c: int(strata_distribution.get(c, 0)) for c in all_combos}
    shortfalls = {
        c: expected_counts[c] - actual_counts[c]
        for c in all_combos
        if expected_counts[c] > actual_counts[c]
    }

    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())
    total_shortfall = sum(shortfalls.values())
    success = total_shortfall == 0

    return {
        "success": success,
        "selected_count": selected_count,
        "candidates_count": candidates_count,
        "per_axis_weights": per_axis_weights,
        "combo_ratios": combo_ratios,
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "shortfalls": shortfalls,
        "total_expected": total_expected,
        "total_actual": total_actual,
        "total_shortfall": total_shortfall,
    }


# -----------------------------------------------------------------------------
# Orchestrace pipeline (run_curation_pipeline)
# -----------------------------------------------------------------------------


def run_curation_pipeline(
    video_path: str,
    out_dir: str,
    config: Optional[Dict] = None,
    **kwargs,
) -> None:
    """
    Hlavní orchestrace kurace snímků — spojení všech kroků do pipeline.

    Pořadí kroků:
      1) Pre-filter & process → získáme kandidáty s metrikami a embeddingy.
      2) ML skórování pomocí MLFrameScorer (vyhodnocení kvality + novosti).
      3) Přiřazení strat (assign_strata_to_frames).
      4) Výběr a deduplikace (select_and_deduplicate).
      5) Uložení výsledků a manifest (save_results).

    Args:
        video_path (str): Cesta k video souboru.
        out_dir (str): Výstupní adresář.
        config (Optional[Dict]): Načtená YAML konfigurace.
        **kwargs: Další parametry (stride, target_size, min_sharpness, min_contrast,
                novelty_threshold, dedup_method, manifest_name, run_params apod.).
    """
    run_params = kwargs.get("run_params", {})
    logger.info(f"Running pipeline with params: {run_params}")

    # --- Krok 1: Pre-filter & compute metrics / embeddings ---
    candidates = prefilter_and_process_frames(
        video_path=video_path,
        stride=kwargs.get("stride") or 1,
        min_sharpness=kwargs.get("min_sharpness") or 80.0,
        min_contrast=kwargs.get("min_contrast") or 20.0,
        config=config,
    )
    if not candidates:
        logger.warning("No candidates after pre-filter. Exiting pipeline.")
        return

    # --- Krok 2: ML skórování (quality + novelty) ---
    logger.info("Starting ML scoring...")
    scorer = MLFrameScorer(
        novelty_threshold=kwargs.get("novelty_threshold") or 0.3, config=config
    )
    scorer.score(candidates, config=config)
    logger.info("ML scoring completed.")

    # --- Krok 3: Přiřazení strat (binned agro-proxy) ---
    assign_strata_to_frames(candidates, config=config)

    # --- Krok 4: Výběr a deduplikace ---
    final_frames = select_and_deduplicate(
        frames=candidates,
        target_size=kwargs.get("target_size"),
        dedup_method=kwargs.get("dedup_method"),
        config=config,
    )

    # --- Krok 5: Uložení výsledků a manifest ---
    save_results(
        frames=final_frames,
        out_dir=out_dir,
        video_path=video_path,
        num_candidates=len(candidates),
        run_params=run_params,
        manifest_name=kwargs.get("manifest_name") or "manifest.json",
        config=config,
    )

    logger.info("Frame curation pipeline completed successfully.")


# -----------------------------------------------------------------------------
# Pomocné utilitky: načítání YAML, čtení/pretty print manifest statistik
# -----------------------------------------------------------------------------


def load_yaml(path: Optional[str]) -> Dict:
    """Načte YAML konfigurační soubor a vrátí dict (pokud existuje)."""
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML file {path}: {e}")
    return {}


def print_human_readable_statistics(
    manifest_path: str,
    elapsed: float = None,
) -> None:
    """
    Vytiskne přehledné statistiky z `manifest.json`.

    Vypisuje:
      - základní info (video, output)
      - počty kandidátů a vybraných snímků + selection ratio
      - agregované ML skóre (avg, median, range)
      - rozdělení napříč osami stratifikace (altitude, view, cover, lighting)
      - volitelně dobu zpracování
    """
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read manifest {manifest_path}: {e}")
        return

    print("\n" + "=" * 80)
    print("📊 FRAME CURATION STATISTICS")
    print("=" * 80)

    task_summary = manifest.get("task_summary", {})
    print(f"🎬 Video: {task_summary.get('video', 'Unknown')}")
    print(f"📁 Output: {task_summary.get('output_dir', 'Unknown')}")
    print(
        f"🔢 Selection: {task_summary.get('selected_count', 'N/A')} frames selected from {task_summary.get('candidates_count', 'N/A')} candidates"
    )
    print(f"📈 Selection ratio: {task_summary.get('selection_ratio', 'N/A')}")

    run_params = manifest.get("run_params", {})
    if run_params:
        print("\n⚙️ RUN PARAMETERS:")
        order = [
            "video",
            "out",
            "config",
            "stride",
            "target_size",
            "min_sharpness",
            "min_contrast",
            "novelty_threshold",
            "dedup_method",
            "manifest",
            "debug",
        ]
        for key in order:
            if key in run_params:
                param = run_params[key]
                value = param.get("value", "N/A")
                source = param.get("source", "N/A")
                print(f"   - {key:20}: {str(value):<25} (source: {source})")

    frames = manifest.get("frames", [])
    if not frames:
        print("\n⚠️ No frames in manifest.")
        print("=" * 80)
        return

    ml_scores = [fr["ml_score"] for fr in frames if "ml_score" in fr]
    if ml_scores:
        print(f"\n🎯 ML SCORE (quality + novelty):")
        print(f"   Avg: {np.mean(ml_scores):.3f}")
        print(f"   Median: {np.median(ml_scores):.3f}")
        print(f"   Range: {min(ml_scores):.3f} - {max(ml_scores):.3f}")

    # Prefer top-level axes_summary in manifest.task_summary if available
    axes_summary = manifest.get("task_summary", {}).get("axes_summary")
    if axes_summary:
        axis_counts = {
            axis: defaultdict(int, counts) for axis, counts in axes_summary.items()
        }
    else:
        axis_counts = {
            "altitude": defaultdict(int),
            "view": defaultdict(int),
            "cover": defaultdict(int),
            "lighting": defaultdict(int),
        }
        for fr in frames:
            strata = fr.get("strata", [])
            if len(strata) == 4:
                a, v, c, l = strata
                axis_counts["altitude"][a] += 1
                axis_counts["view"][v] += 1
                axis_counts["cover"][c] += 1
                axis_counts["lighting"][l] += 1

    print("\n🏔️ STRATIFICATION AXES DISTRIBUTION:")
    total_selected = task_summary.get("selected_count", 0)
    for axis, counts in axis_counts.items():
        print(f"   --- {axis.upper()} ---")
        for label, count in sorted(counts.items()):
            pct = (count / total_selected * 100) if total_selected > 0 else 0
            print(f"     {label:10}: {count:4} ({pct:5.1f}%)")

    # Targets evaluation (if present)
    targets_eval = (
        manifest.get("targets_evaluation") if "manifest" in locals() else None
    )
    # Fallback: try to load from manifest_path file if not in local manifest variable
    if not targets_eval:
        # attempt to read targets_evaluation from manifest file path if exists
        try:
            with open(manifest_path, "r", encoding="utf-8") as _fh:
                _m = json.load(_fh)
                targets_eval = _m.get("targets_evaluation")
        except Exception:
            targets_eval = None

    if targets_eval:
        print("\n🎯 STRATIFICATION TARGETS EVALUATION:")
        print(
            f"   Selected frames: {targets_eval.get('selected_count')} (candidates: {targets_eval.get('candidates_count')})"
        )
        success = targets_eval.get("success", False)
        if success:
            print("   ✅ Targets satisfied.")
        else:
            total_shortfall = targets_eval.get("total_shortfall", 0)
            print("   ❌ Targets NOT satisfied.")
            print(f"   Total shortfall (frames needed): {total_shortfall}")
            shortfalls = targets_eval.get("shortfalls", {})
            if shortfalls:
                print("   Missing by combination:")
                for combo, deficit in sorted(shortfalls.items(), key=lambda x: -x[1]):
                    expected = targets_eval.get("expected_counts", {}).get(combo, 0)
                    actual = targets_eval.get("actual_counts", {}).get(combo, 0)
                    pct = (deficit / expected * 100) if expected > 0 else 0
                    print(
                        f"     - {combo}: expected {expected}, actual {actual}, missing {deficit} ({pct:4.1f}%)"
                    )

            # Recommendations for next run
            print("\n   Suggested actions to improve coverage in next run:")
            print(
                "     - Increase --target-size by the total shortfall to attempt collecting more frames."
            )
            print(
                "     - Reduce --min-sharpness / --min-contrast to allow more candidates pass prefilter."
            )
            print(
                "     - Reduce --stride to sample more frames (e.g., halve the value)."
            )
            print(
                "     - Relax deduplication strictness (if using greedy: lower cosine_threshold, or switch to dbscan and increase eps)."
            )
            print(
                "     - Lower --novelty-threshold to accept more non-novel frames, or provide additional input videos to increase candidate pool."
            )

    if elapsed is not None:
        mins, secs = divmod(int(elapsed), 60)
        print(f"\n⏱️ Elapsed time: {mins}m {secs}s")

    print("=" * 80)


# -----------------------------------------------------------------------------
# CLI / entrypoint
# -----------------------------------------------------------------------------


def setup_logging(log_dir: str, debug: bool):
    """
    Nastaví logging do souboru a volitelně i na konzoli.

    Poznámka: aktuální implementace používá `logging.FileHandler` bez rotace.
    Pokud chcete rotaci logu (doporučeno v produkci), použijte `RotatingFileHandler`
    nebo `TimedRotatingFileHandler` místo FileHandler.
    """
    level = logging.DEBUG if debug else logging.INFO
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "curation.log")

    # File handler (přepisuje/nový soubor na každý běh)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_handler.setFormatter(formatter)

    # Root logger: nahradíme existující handlery aktuálním file handlerem
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]
    root_logger.setLevel(level)

    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def main():
    """Hlavní spouštěcí bod — parsování CLI argumentů a spuštění pipeline."""
    import sys

    # Dočasné parsování pro načtení config cesty (--config)
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", default=None, help="Path to YAML config.")
    temp_args, remaining_argv = temp_parser.parse_known_args()

    # Načtení konfigurace a výchozích hodnot (defaults v YAML)
    config = load_yaml(temp_args.config)
    defaults = config.get("defaults", {})

    # Hlavní parser CLI
    parser = argparse.ArgumentParser(
        description="Orchestrátor pro ML-driven kuraci snímků.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Path to input video file.")
    parser.add_argument("-o", "--out", help="Output directory.")
    parser.add_argument(
        "--config", default=temp_args.config, help="Path to YAML configuration."
    )
    parser.add_argument("--stride", type=int, help="Sample every N-th frame.")
    parser.add_argument(
        "--target-size",
        type=int,
        help="Target number of frames (triggers stratified selection).",
    )
    parser.add_argument(
        "--min-sharpness", type=float, help="Min sharpness (Variance of Laplacian)."
    )
    parser.add_argument(
        "--min-contrast", type=float, help="Min contrast (std dev of grayscale)."
    )
    parser.add_argument(
        "--novelty-threshold", type=float, help="Novelty/prototype threshold (0..1)."
    )
    parser.add_argument(
        "--dedup-method", choices=["greedy", "dbscan"], help="Deduplication method."
    )
    parser.add_argument("--manifest", help="Name of output manifest file.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory without prompting.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging to console."
    )

    # Apply defaults from config (parser.set_defaults)
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    # --- Collect run_params with source (cli | config | default | derived) ---
    cli_dests = set()
    opt_string_to_dest = {
        opt: action.dest for action in parser._actions for opt in action.option_strings
    }
    for arg in sys.argv[1:]:
        key = arg.split("=")[0]
        if key in opt_string_to_dest:
            cli_dests.add(opt_string_to_dest[key])
    if args.video:
        cli_dests.add("video")

    run_params = {}
    for key, value in vars(args).items():
        if key in ["help"]:
            continue
        source = ""
        if key in cli_dests:
            source = "cli"
        elif key in defaults:
            source = "config"
        else:
            source = "default"

        actual_value = value
        if actual_value is None:
            if key in defaults:
                actual_value = defaults[key]
            else:
                hardcoded_defaults = {
                    "stride": 3,
                    "min_sharpness": 80.0,
                    "min_contrast": 20.0,
                    "novelty_threshold": 0.3,
                    "dedup_method": "greedy",
                    "manifest": "manifest.json",
                    "config": None,
                    "target_size": None,
                }
                actual_value = hardcoded_defaults.get(key, None)

        run_params[key] = {"value": actual_value, "source": source}

    # Derive default output directory if not provided
    is_out_derived = not (args.out or "out" in cli_dests or "out" in defaults)
    if is_out_derived or args.out is None:
        video_base = os.path.splitext(os.path.basename(args.video))[0]
        args.out = os.path.join("data", "output", video_base)
        run_params["out"] = {"value": args.out, "source": "derived"}

    # If output directory exists, handle overwrite logic BEFORE initializing logging
    # (avoids creating log files inside an output dir we may immediately remove).
    if os.path.exists(args.out):
        # If explicit overwrite flag, remove without prompting
        if args.overwrite:
            try:
                shutil.rmtree(args.out)
            except Exception as e:
                # Logging isn't configured yet; print a concise error and exit.
                print(
                    f"Error removing existing output directory '{args.out}': {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # If running interactively, ask user for confirmation (default: No)
            if sys.stdin.isatty():
                try:
                    ans = input(
                        f"Output directory '{args.out}' exists. Overwrite? (yes/No): "
                    )
                except EOFError:
                    ans = ""
                if ans.strip().lower() in ("y", "yes"):
                    try:
                        shutil.rmtree(args.out)
                    except Exception as e:
                        print(
                            f"Error removing existing output directory '{args.out}': {e}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                else:
                    # Respect user's choice to not overwrite
                    print(
                        "Output directory exists and overwrite not confirmed. Exiting.",
                        file=sys.stderr,
                    )
                    sys.exit(0)
            else:
                # Non-interactive environment: require explicit --overwrite
                print(
                    f"Output directory '{args.out}' exists and --overwrite not provided; cannot prompt in non-interactive mode.",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Setup logging
    setup_logging(args.out, args.debug)

    logger.info("Starting frame curation process.")
    if args.config:
        logger.info(f"Loaded config from: {args.config}")
    else:
        logger.info("Using defaults and CLI arguments.")

    # Run pipeline with timing
    start_time = time.time()
    run_curation_pipeline(
        video_path=args.video,
        out_dir=args.out,
        config=config,
        run_params=run_params,
        stride=args.stride,
        target_size=args.target_size,
        min_sharpness=args.min_sharpness,
        min_contrast=args.min_contrast,
        novelty_threshold=args.novelty_threshold,
        dedup_method=args.dedup_method,
        manifest_name=args.manifest,
    )
    elapsed = time.time() - start_time

    # Print human-readable statistics from manifest (if exists)
    manifest_path = os.path.join(args.out, (args.manifest or "manifest.json"))
    if os.path.exists(manifest_path):
        print_human_readable_statistics(manifest_path, elapsed=elapsed)
    else:
        logger.warning(f"Manifest file not found: {manifest_path}")


if __name__ == "__main__":
    main()
