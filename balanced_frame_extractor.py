#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-Driven Frame Curation — Agro Stratification
==============================================

Tento skript slouží jako orchestrátor pro kuraci snímků z videí.
Používá knihovnu `bfe` k provedení následujících kroků:

1.  **Prefilter snímků**: Odstraní snímky s nízkou ostrostí a kontrastem.
2.  **Výpočet metrik a embeddingů**: Pro každý snímek spočítá metriky kvality,
    agro-proxy a embeddingy.
3.  **ML skórování**: Ohodnotí každý snímek na základě kvality a novosti obsahu.
4.  **Stratifikace**: Rozdělí snímky do kategorií (strat) podle agro-proxy metrik.
5.  **Výběr snímků**: Vybere nejlepší snímky na základě stratifikace a ML skóre.
6.  **Deduplikace**: Odstraní vizuálně podobné snímky.
7.  **Uložení výsledků**: Uloží vybrané snímky a vytvoří `manifest.json` se statistikami.

Použití:
    python balanced_frame_extractor.py <video_path> -o <output_dir> --config config.yaml
"""
from __future__ import annotations

import os
import json
import argparse
import logging
import time
from collections import defaultdict
from typing import List, Dict, Optional

import cv2
import numpy as np
import yaml

# Importy z refaktorované knihovny `bfe`
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
# Krok 1: Prefilter a výpočet metrik
# -----------------------------------------------------------------------------


def prefilter_and_process_frames(
    video_path: str,
    stride: int = 1,
    min_sharpness: float = 80.0,
    min_contrast: float = 20.0,
    config: Optional[Dict] = None,
) -> List[FrameInfo]:
    """
    Iteruje video, provádí pre-filtraci a počítá všechny potřebné metriky a embeddingy.
    Tato funkce kombinuje původní `prefilter_candidates` s výpočty, které následovaly.

    Args:
        video_path (str): Cesta k videu.
        stride (int): Krok pro čtení snímků.
        min_sharpness (float): Minimální povolená ostrost.
        min_contrast (float): Minimální povolený kontrast.
        config (Optional[Dict]): Konfigurační slovník.

    Returns:
        List[FrameInfo]: Seznam kandidátských snímků s vyplněnými metrikami.
    """
    logger.info(f"Zahájení pre-filtrace a zpracování pro video: {video_path}")
    logger.info(
        f"Parametry: krok={stride}, min_ostrost={min_sharpness}, min_kontrast={min_contrast}"
    )

    candidates: List[FrameInfo] = []
    total_frames = 0
    filtered_out = 0

    # Načtení konfiguračních hodnot
    output_config = (config or {}).get("output", {})
    log_intervals = output_config.get("log_intervals", {})
    frames_processed_interval = log_intervals.get("frames_processed", 1000)
    proxies_config = (config or {}).get("proxies", {})
    view_bins = proxies_config.get("view_entropy_bins", 8)
    green_threshold = proxies_config.get("green_cover_threshold", 0.6)

    for idx, t_sec, bgr in iter_video_frames(video_path, stride=stride, config=config):
        total_frames += 1
        if total_frames % frames_processed_interval == 0:
            logger.debug(
                f"Zpracováno {total_frames} snímků, ponecháno {len(candidates)} kandidátů"
            )

        # Základní metriky kvality
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)
        contrast = estimate_contrast(gray)

        # Prefilter
        if sharpness < min_sharpness or contrast < min_contrast:
            filtered_out += 1
            logger.debug(
                f"Snímek {idx} odfiltrován: ostrost={sharpness:.1f} (min={min_sharpness}), "
                f"kontrast={contrast:.1f} (min={min_contrast})"
            )
            continue

        # Pokročilé metriky kvality
        mean, under, over = exposure_metrics(gray, config=config)
        exposure_score = exposure_score_from_metrics(mean, under, over, config=config)
        noise_score = estimate_noise_score(gray, config=config)

        # Embeddingy
        hsv_hist, lowres_vec, embed = combined_embed(bgr, config=config)

        # Agro proxies
        hf_energy = altitude_proxy(gray, config=config)
        view_entropy_val = view_entropy(gray, bins=view_bins, config=config)
        green_cover = green_cover_ratio(bgr, threshold=green_threshold, config=config)
        lighting_mean = float(np.mean(gray))

        logger.debug(
            f"Snímek {idx}: ostrost={sharpness:.1f}, kontrast={contrast:.1f}, exp_skóre={exposure_score:.3f}, "
            f"hf_energie={hf_energy:.2f}, entropie_pohledu={view_entropy_val:.2f}, zelen={green_cover:.3f}"
        )

        # Vytvoření a uložení objektu FrameInfo
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
        f"Prefilter dokončen: Z {total_frames} snímků ponecháno {len(candidates)} kandidátů, "
        f"{filtered_out} odfiltrováno."
    )
    return candidates


# -----------------------------------------------------------------------------
# Krok 2: Binning (kategorizace) pro stratifikaci
# -----------------------------------------------------------------------------


def assign_strata_to_frames(
    frames: List[FrameInfo], config: Optional[Dict] = None
) -> None:
    """
    Přiřadí každému snímku jeho stratum (kategorii) na základě agro-proxy metrik.

    Args:
        frames (List[FrameInfo]): Seznam snímků ke zpracování.
        config (Optional[Dict]): Konfigurační slovník.
    """
    if not frames:
        return

    logger.info("Zahájení výpočtu a přiřazování strat...")

    # Získání prahových hodnot z konfigurace
    stratification_config = (config or {}).get("stratification", {})
    thresholds = stratification_config.get("thresholds", {})
    view_threshold = thresholds.get("view_entropy", 1.8)
    cover_threshold = thresholds.get("cover_ratio", 0.5)
    lighting_threshold = thresholds.get("lighting_mean", 115)

    # Pro `altitude` potřebujeme kvantil, počítáme ho z celého datasetu
    hf_vals = np.array([f.hf_energy for f in frames], dtype=np.float32)
    altitude_q50 = float(np.quantile(hf_vals, 0.5))
    logger.debug(f"Kvantil pro altitude (q50 z HF energie): {altitude_q50:.2f}")

    strata_counts = defaultdict(int)
    for f in frames:
        a = bin_altitude(f.hf_energy, altitude_q50)
        v = bin_view(f.view_entropy_val, t=view_threshold)
        c = bin_cover(f.green_cover, threshold=cover_threshold)
        l = bin_lighting(f.lighting_mean, threshold=lighting_threshold)
        f.strata = (a, v, c, l)
        strata_counts[f.strata] += 1

    logger.debug(f"Distribuce snímků ve stratách: {dict(strata_counts)}")
    logger.info("Přiřazování strat dokončeno.")


# -----------------------------------------------------------------------------
# Krok 3: Výběr a deduplikace
# -----------------------------------------------------------------------------


def select_and_deduplicate(
    frames: List[FrameInfo],
    target_size: Optional[int],
    config: Optional[Dict] = None,
    dedup_method: Optional[str] = None,
) -> List[FrameInfo]:
    """
    Provede finální výběr snímků pomocí stratifikace (pokud je `target_size` zadán)
    nebo prahování, a následně provede deduplikaci.

    Args:
        frames (List[FrameInfo]): Seznam kandidátských snímků, seřazených dle ML skóre.
        target_size (Optional[int]): Cílový počet snímků.
        config (Optional[Dict]): Konfigurační slovník.
        dedup_method (Optional[str]): Metoda deduplikace ('greedy' nebo 'dbscan').

    Returns:
        List[FrameInfo]: Finální seznam vybraných snímků.
    """
    if not frames:
        return []

    # Seřazení kandidátů podle ML skóre (od nejlepšího)
    frames_sorted = sorted(frames, key=lambda x: x.ml_score, reverse=True)

    # --- Předběžný výběr ---
    if target_size is None:
        # Metoda založená na prahu: vezmeme top N % kandidátů
        selection_config = (config or {}).get("selection", {})
        threshold_ratio = selection_config.get("threshold_selection_ratio", 0.25)
        threshold_count = max(1, int(len(frames) * threshold_ratio))
        logger.info(
            f"Výběr na základě prahu: vybráno top {threshold_count} snímků ({threshold_ratio:.1%} kandidátů)."
        )
        preliminary_selection = frames_sorted[:threshold_count]
    else:
        # Stratifikovaný výběr pro dosažení cílového počtu
        logger.info(f"Výběr pomocí stratifikace s cílem {target_size} snímků.")
        stratifier = AgroStratifier(config or {})
        preliminary_selection = stratifier.select(
            frames_sorted, target_size, config=config
        )
        logger.debug(f"Stratifikovaný výběr: {len(preliminary_selection)} snímků.")

        # Doplnění o nejkvalitnější snímky, pokud stratifikace vrátila méně, než je cíl
        if len(preliminary_selection) < target_size:
            needed = target_size - len(preliminary_selection)
            remaining_candidates = [
                f for f in frames_sorted if f not in preliminary_selection
            ]
            to_add = remaining_candidates[:needed]
            preliminary_selection.extend(to_add)
            logger.debug(f"Doplněno {len(to_add)} snímky s nejvyšším skóre.")

    logger.info(f"Předběžný výběr obsahuje {len(preliminary_selection)} snímků.")

    # --- Deduplikace ---
    logger.info("Zahájení deduplikace...")
    deduplication_config = (config or {}).get("deduplication", {})
    method = (dedup_method or deduplication_config.get("method") or "greedy").lower()

    if method == "dbscan":
        eps = deduplication_config.get("eps", None)
        min_samples = int(deduplication_config.get("min_samples", 1))
        logger.info(
            f"Používá se DBSCAN deduplikace (eps={eps}, min_samples={min_samples})."
        )
        final_selection = deduplicate_dbscan(
            preliminary_selection, eps=eps, min_samples=min_samples, config=config
        )
    else:
        cosine_threshold = deduplication_config.get("cosine_threshold", 0.85)
        logger.info(
            f"Používá se 'greedy' deduplikace (práh kosinové podobnosti={cosine_threshold})."
        )
        final_selection = deduplicate_quality_first(
            preliminary_selection, cosine_threshold=cosine_threshold, config=config
        )

    logger.info(f"Po deduplikaci zbylo {len(final_selection)} snímků.")

    # --- Doplnění po deduplikaci (pokud je to nutné a možné) ---
    if target_size is not None and len(final_selection) < target_size:
        needed = target_size - len(final_selection)
        logger.debug(
            f"Počet snímků ({len(final_selection)}) je pod cílem ({target_size}). Pokus o doplnění {needed} snímků."
        )

        # Vytvoříme pool zbývajících kandidátů, kteří nebyli vybráni
        remaining_pool = [f for f in frames_sorted if f not in final_selection]
        to_add = remaining_pool[:needed]

        if to_add:
            final_selection.extend(to_add)
            logger.debug(
                f"Doplněno {len(to_add)} snímků, finální počet: {len(final_selection)}."
            )

    # Finální seřazení a oříznutí na cílovou velikost
    final_selection = sorted(final_selection, key=lambda x: x.ml_score, reverse=True)
    if target_size is not None:
        final_selection = final_selection[:target_size]

    return final_selection


# -----------------------------------------------------------------------------
# Ukládání výsledků
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
    Uloží vybrané snímky na disk a vygeneruje manifest.json se všemi metadaty.

    Args:
        frames (List[FrameInfo]): Finální seznam vybraných snímků.
        out_dir (str): Výstupní adresář.
        video_path (str): Cesta k původnímu videu.
        num_candidates (int): Počet kandidátů po pre-filtraci.
        run_params (Dict): Parametry běhu skriptu.
        manifest_name (str): Název souboru manifestu.
        config (Optional[Dict]): Konfigurační slovník.
    """
    if not frames:
        logger.warning("Žádné snímky k uložení.")
        return

    logger.info(f"Ukládání {len(frames)} vybraných snímků do adresáře: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    output_config = (config or {}).get("output", {})
    jpeg_quality = output_config.get("jpeg_quality", 95)
    log_interval = output_config.get("log_intervals", {}).get("frames_saved", 100)

    saved_paths = []
    for i, f in enumerate(frames):
        fname = f"frame_{i:06d}_src{f.idx:06d}_t{f.t_sec:010.3f}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, f.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        saved_paths.append(path)
        if (i + 1) % log_interval == 0:
            logger.debug(f"Uloženo {i + 1}/{len(frames)} snímků")

    # Sestavení manifestu + agregace distribuce strat pro celou kolekci
    # Vytvoříme dvě agregace:
    #  - strata_distribution: mapuje kombinovanou klíčovou string → počet
    #  - axes_summary: pro každou osu (altitude, view, cover, lighting) map hodnot → počet
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
    }

    manifest = {
        "video": os.path.abspath(video_path),
        "run_params": run_params or {},
        "count_candidates": num_candidates,
        "count_selected": len(frames),
        "out_dir": os.path.abspath(out_dir),
        "axes": ["altitude", "view", "cover", "lighting"],
        "strata_distribution": strata_distribution,
        "axes_summary": axes_summary,
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

    manifest_path = os.path.join(out_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    logger.info(f"Manifest uložen do: {manifest_path}")


# -----------------------------------------------------------------------------
# Hlavní orchestrační funkce
# -----------------------------------------------------------------------------


def run_curation_pipeline(
    video_path: str,
    out_dir: str,
    config: Optional[Dict] = None,
    **kwargs,
) -> None:
    """
    Hlavní funkce, která orchestruje celý proces kurace snímků.

    Args:
        video_path (str): Cesta k video souboru.
        out_dir (str): Výstupní adresář.
        config (Optional[Dict]): Načtená YAML konfigurace.
        **kwargs: Další parametry z CLI (stride, target_size, atd.).
    """
    run_params = kwargs.get("run_params", {})
    logger.info(f"Spouštění pipeline s parametry: {run_params}")

    # --- Krok 1: Prefilter a výpočet metrik ---
    candidates = prefilter_and_process_frames(
        video_path=video_path,
        stride=kwargs.get("stride") or 1,
        min_sharpness=kwargs.get("min_sharpness") or 80.0,
        min_contrast=kwargs.get("min_contrast") or 20.0,
        config=config,
    )
    if not candidates:
        logger.warning("Po pre-filtraci nezbyli žádní kandidáti. Proces končí.")
        return

    # --- Krok 2: ML Skórování ---
    logger.info("Zahájení ML skórování...")
    scorer = MLFrameScorer(
        novelty_threshold=kwargs.get("novelty_threshold") or 0.3, config=config
    )
    scorer.score(candidates, config=config)
    logger.info("ML skórování dokončeno.")

    # --- Krok 3: Přiřazení strat ---
    assign_strata_to_frames(candidates, config=config)

    # --- Krok 4: Výběr a deduplikace ---
    final_frames = select_and_deduplicate(
        frames=candidates,
        target_size=kwargs.get("target_size"),
        dedup_method=kwargs.get("dedup_method"),
        config=config,
    )

    # --- Krok 5: Uložení výsledků ---
    save_results(
        frames=final_frames,
        out_dir=out_dir,
        video_path=video_path,
        num_candidates=len(candidates),
        run_params=run_params,
        manifest_name=kwargs.get("manifest_name") or "manifest.json",
        config=config,
    )

    logger.info("Pipeline na kuraci snímků byla úspěšně dokončena.")


# -----------------------------------------------------------------------------
# Pomocné funkce pro CLI a statistiky
# -----------------------------------------------------------------------------


def load_yaml(path: Optional[str]) -> Dict:
    """Načte YAML soubor, pokud existuje."""
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Chyba při načítání YAML souboru {path}: {e}")
    return {}


def print_human_readable_statistics(
    manifest_path: str,
    elapsed: float = None,
) -> None:
    """Vytiskne přehledné statistiky z manifest.json souboru."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.error(f"Nepodařilo se načíst manifest {manifest_path}: {e}")
        return

    print("\n" + "=" * 80)
    print("📊 STATISTIKY KURACE SNÍMKŮ")
    print("=" * 80)

    # Základní informace z task_summary
    task_summary = manifest.get("task_summary", {})
    print(f"🎬 Video: {task_summary.get('video', 'Neznámé')}")
    print(f"📁 Výstup: {task_summary.get('output_dir', 'Neznámý')}")
    print(
        f"🔢 Výběr: {task_summary.get('selected_count', 'N/A')} snímků vybráno z {task_summary.get('candidates_count', 'N/A')} kandidátů"
    )
    print(f"📈 Poměr výběru: {task_summary.get('selection_ratio', 'N/A')}")

    # Parametry běhu
    run_params = manifest.get("run_params", {})
    if run_params:
        print("\n⚙️ POUŽITÉ PARAMETRY:")
        # Pořadí pro hezčí výpis
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
                print(f"   - {key:20}: {str(value):<25} (zdroj: {source})")

    frames = manifest.get("frames", [])
    if not frames:
        print("\n⚠️ V manifestu nebyly nalezeny žádné snímky.")
        print("=" * 80)
        return

    # Statistiky ML skóre
    ml_scores = [fr["ml_score"] for fr in frames if "ml_score" in fr]
    if ml_scores:
        print(f"\n🎯 ML SKÓRE KVALITY:")
        print(f"   Průměr: {np.mean(ml_scores):.3f}")
        print(f"   Medián: {np.median(ml_scores):.3f}")
        print(f"   Rozsah: {min(ml_scores):.3f} - {max(ml_scores):.3f}")

    # Rozdělení podle strat — preferovat top-level agregace z manifestu pokud jsou dostupné
    axes_summary = manifest.get("axes_summary")
    if axes_summary:
        # axes_summary je map: osa -> {hodnota: count}
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

    print("\n🏔️ ROZDĚLENÍ PODLE OS STRATIFIKACE:")
    total_selected = task_summary.get("selected_count", 0)
    for axis, counts in axis_counts.items():
        print(f"   --- {axis.upper()} ---")
        for label, count in sorted(counts.items()):
            pct = (count / total_selected * 100) if total_selected > 0 else 0
            print(f"     {label:10}: {count:4} ({pct:5.1f}%)")

    if elapsed is not None:
        mins, secs = divmod(int(elapsed), 60)
        print(f"\n⏱️ Doba zpracování: {mins}m {secs}s")

    print("=" * 80)


# -----------------------------------------------------------------------------
# CLI a spouštěcí bod
# -----------------------------------------------------------------------------


def setup_logging(log_dir: str, debug: bool):
    """Nastaví logování do souboru a volitelně na konzoli."""
    level = logging.DEBUG if debug else logging.INFO
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "curation.log")

    # Vytvoření handleru pro soubor s rotací
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_handler.setFormatter(formatter)

    # Nastavení root loggeru
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]  # Odebrání existujících handlerů
    root_logger.setLevel(level)

    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def main():
    """Hlavní spouštěcí funkce, parsuje argumenty a spouští pipeline."""
    import sys

    # Dočasné parsování pro načtení konfiguračního souboru
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", default=None, help="Cesta k YAML konfiguraci.")
    temp_args, remaining_argv = temp_parser.parse_known_args()

    # Načtení konfigurace a výchozích hodnot
    config = load_yaml(temp_args.config)
    defaults = config.get("defaults", {})

    # Hlavní parser argumentů
    parser = argparse.ArgumentParser(
        description="Orchestrátor pro ML-driven kuraci snímků.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Cesta ke vstupnímu video souboru.")
    parser.add_argument("-o", "--out", help="Cesta k výstupnímu adresáři.")
    parser.add_argument(
        "--config", default=temp_args.config, help="Cesta k YAML konfiguraci."
    )
    parser.add_argument("--stride", type=int, help="Vybírat každý N-tý snímek.")
    parser.add_argument(
        "--target-size", type=int, help="Cílový počet snímků (potlačí výběr prahem)."
    )
    parser.add_argument(
        "--min-sharpness", type=float, help="Minimální ostrost (Variance of Laplacian)."
    )
    parser.add_argument(
        "--min-contrast", type=float, help="Minimální kontrast (std dev šedotónu)."
    )
    parser.add_argument(
        "--novelty-threshold", type=float, help="Práh pro prototypy novosti (0..1)."
    )
    parser.add_argument(
        "--dedup-method", choices=["greedy", "dbscan"], help="Metoda deduplikace."
    )
    parser.add_argument("--manifest", help="Název výstupního manifest souboru.")
    parser.add_argument(
        "--debug", action="store_true", help="Zapnout debugovací výpisy."
    )

    # Nastavení výchozích hodnot z configu
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    # --- Sběr parametrů běhu a jejich zdrojů ---
    cli_dests = set()
    opt_string_to_dest = {
        opt: action.dest for action in parser._actions for opt in action.option_strings
    }
    for arg in sys.argv[1:]:
        key = arg.split("=")[0]
        if key in opt_string_to_dest:
            cli_dests.add(opt_string_to_dest[key])
    if args.video:  # Pozicni argument
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

        # Určení skutečné hodnoty parametru
        actual_value = value
        if actual_value is None:
            if key in defaults:
                actual_value = defaults[key]
            else:
                # Hardcoded výchozí hodnoty pro parametry bez config hodnot
                hardcoded_defaults = {
                    "stride": 1,
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

    # Výchozí výstupní adresář, pokud není zadán
    is_out_derived = not (args.out or "out" in cli_dests or "out" in defaults)
    if is_out_derived or args.out is None:
        video_base = os.path.splitext(os.path.basename(args.video))[0]
        args.out = os.path.join("data", "output", video_base)
        run_params["out"] = {"value": args.out, "source": "derived"}

    # Nastavení logování
    setup_logging(args.out, args.debug)

    logger.info("Zahájení procesu kurace snímků.")
    if args.config:
        logger.info(f"Načtena konfigurace z: {args.config}")
    else:
        logger.info("Používá se výchozí konfigurace a CLI argumenty.")

    # Spuštění pipeline s časováním
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

    # Tisk statistik
    manifest_path = os.path.join(args.out, (args.manifest or "manifest.json"))
    if os.path.exists(manifest_path):
        print_human_readable_statistics(manifest_path, elapsed=elapsed)
    else:
        logger.warning(f"Soubor manifestu nebyl nalezen: {manifest_path}")


if __name__ == "__main__":
    main()
