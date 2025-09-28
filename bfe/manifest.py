#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manifest.py
===========
Funkce pro ukládání výsledků a generování manifestu.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Dict, Optional

import cv2
import numpy as np

from bfe.frame_info import FrameInfo
from bfe.stratification import evaluate_targets_from_config

logger = logging.getLogger(__name__)


def save_manifest_and_frames(
    frames: List[FrameInfo],
    out_dir: str,
    video_path: str,
    num_candidates: int,
    run_params: Dict,
    manifest_name: str = "manifest.json",
    config: Optional[Dict] = None,
    elapsed_time: Optional[float] = None,
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
        elapsed_time (Optional[float]): Celkový čas běhu pipeline.
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
        "elapsed_time_sec": elapsed_time,
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
