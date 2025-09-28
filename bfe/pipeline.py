#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Hlavní orchestrace pipeline pro kuraci snímků.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Dict, Optional

import cv2
import numpy as np

from bfe.binning import bin_altitude, bin_cover, bin_lighting, bin_view
from bfe.deduplication import deduplicate_dbscan, deduplicate_quality_first
from bfe.embeddings import combined_embed
from bfe.frame_info import FrameInfo
from bfe.manifest import save_manifest_and_frames
from bfe.proxies import altitude_proxy, green_cover_ratio, view_entropy
from bfe.quality_metrics import (
    estimate_contrast,
    estimate_noise_score,
    exposure_metrics,
    exposure_score_from_metrics,
    variance_of_laplacian,
)
from bfe.scoring import MLFrameScorer
from bfe.stratification import AgroStratifier
from bfe.video_io import iter_video_frames

logger = logging.getLogger(__name__)


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
    save_manifest_and_frames(
        frames=final_frames,
        out_dir=out_dir,
        video_path=video_path,
        num_candidates=len(candidates),
        run_params=run_params,
        manifest_name=kwargs.get("manifest_name") or "manifest.json",
        config=config,
    )

    logger.info("Frame curation pipeline completed successfully.")
