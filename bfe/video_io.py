#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_io.py
===========
Tento modul poskytuje funkce pro čtení snímků z video souborů.
"""
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Dict
import cv2
import numpy as np

def iter_video_frames(
    video_path: str, stride: int = 1, config: Optional[Dict] = None
) -> Iterable[Tuple[int, float, np.ndarray]]:
    """
    Iteruje přes snímky ve videu a vrací je s daným krokem (stride).

    Args:
        video_path (str): Cesta k video souboru.
        stride (int): Vybírat každý N-tý snímek.
        config (Optional[Dict]): Konfigurační slovník (pro fallback FPS).

    Yields:
        Tuple[int, float, np.ndarray]: (index snímku, čas v sekundách, BGR snímek)
    """
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
