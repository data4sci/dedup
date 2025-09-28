#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced Frame Extractor — Agro ML Dataset Curation (orchestrator)
================================================================

Cílem tohoto modulu je orchestrace procesu kurace snímků z droních videí
pro vytvoření vyváženého datasetu pro ML trénink. Implementované kroky
odpovídají popisu v README.md a používají jednotnou terminologii.

- Tento soubor slouží jako CLI entrypoint a volá `run_curation_pipeline` z modulu `bfe.pipeline`.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time

from bfe.pipeline import run_curation_pipeline
from bfe.reporting import print_human_readable_statistics
from bfe.utils import load_yaml, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Hlavní spouštěcí bod — parsování CLI argumentů a spuštění pipeline."""
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
