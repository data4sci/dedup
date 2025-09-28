#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
Pomocné funkce pro načítání konfigurace, nastavení logování a další utility.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

def load_yaml(path: Optional[str]) -> Dict:
    """Načte YAML konfigurační soubor a vrátí dict (pokud existuje)."""
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML file {path}: {e}")
    return {}


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

