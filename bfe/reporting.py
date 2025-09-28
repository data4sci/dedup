#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting.py
============
Funkce pro generování a tisk statistik a reportů.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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
    print("📊 FRAME CURATION STATISTICS:")

    task_summary = manifest.get("task_summary", {})
    print(f"🎬 Video: {task_summary.get('video', 'Unknown')}")
    print(f"📁 Output: {task_summary.get('output_dir', 'data/output/Unknown')}")
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

    # --- STRATIFICATION TABLE ---
    print("\n🏔️ STRATIFICATION:")
    total_selected = task_summary.get("selected_count", 0)
    # Prepare targets from config
    strat_cfg = manifest.get("config", {}).get("stratification", {})
    targets_axes = strat_cfg.get("targets_axes", {})
    # Fallback: try to get from config.yaml if not in manifest
    if not targets_axes:
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                import yaml

                cfg = yaml.safe_load(f)
                targets_axes = cfg.get("stratification", {}).get("targets_axes", {})
        except Exception:
            targets_axes = {}

    # Prepare per-value satisfaction (strict: actual >= target with minimal rounding tolerance)
    def satisfied(actual, target):
        return actual >= (target - 0.001)  # 0.1% tolerance for rounding errors

    print(f"{ 'axis':<10} {'value':<10} {'actual':>8} {'target':>8} {'satisfied':>10}")
    print("-" * 50)

    # Get all possible axis values from config axes (to show missing values as 0.0%)
    strat_cfg_axes = {}
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            import yaml

            cfg = yaml.safe_load(f)
            strat_cfg_axes = cfg.get("stratification", {}).get("axes", {})
    except Exception:
        pass

    all_satisfied = True
    for axis in ["altitude", "view", "cover", "lighting"]:
        axis_targets = targets_axes.get(axis, {})
        counts = axis_counts.get(axis, {})
        all_possible_values = strat_cfg_axes.get(axis, list(counts.keys()))

        for label in sorted(all_possible_values):
            actual_count = counts.get(label, 0)
            actual_pct = (
                (actual_count / total_selected * 100) if total_selected > 0 else 0.0
            )
            target_pct = float(
                axis_targets.get(label, 1.0 / max(1, len(all_possible_values))) * 100
            )
            is_satisfied = satisfied(actual_pct / 100, target_pct / 100)
            if not is_satisfied:
                all_satisfied = False
            print(
                f"{axis:<10} {label:<10} {actual_pct:8.1f} {target_pct:8.1f} {'✅' if is_satisfied else '❌':>10}"
            )
    print("-" * 50)

    # Overall satisfaction summary
    print(f"Overall targets: {'✅ Satisfied' if all_satisfied else '❌ NOT satisfied'}")

    # --- Check if target-size was reached ---
    target_size_info = run_params.get("target_size", {})
    target_size = target_size_info.get("value")
    target_size_reached = True
    if target_size is not None:
        selected_count = task_summary.get("selected_count", 0)
        if selected_count < target_size:
            target_size_reached = False
            print(
                f"\n⚠️ WARNING: Target size not reached ({selected_count}/{target_size} frames)."
            )

    # Show recommendations if targets not satisfied or target size not reached
    if not all_satisfied or not target_size_reached:
        print("\n💡 RECOMMENDATIONS:")
        if not all_satisfied:
            print(
                "   - Stratification targets were not met. The pool of candidates might be too small or skewed."
            )
        if not target_size_reached:
            print(
                f"   - The final number of frames ({total_selected}) is less than the requested --target-size ({target_size})."
            )

        print("\n   To improve results, consider the following adjustments:")
        print(
            "   - Loosen pre-filtering: reduce --min-sharpness or --min-contrast to get more candidates."
        )
        print(
            "   - Sample more densely: reduce --stride to process more frames from the video."
        )
        print(
            "   - Adjust novelty scoring: lower --novelty-threshold to be less strict about what is a 'new' frame."
        )
        print(
            "   - Relax deduplication: lower the 'cosine_threshold' in your config to keep more unique frames."
        )

    if elapsed is not None:
        mins, secs = divmod(int(elapsed), 60)
        print(f"\n⏱️ Elapsed time: {mins}m {secs}s")

    print("=" * 80)
