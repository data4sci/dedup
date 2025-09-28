#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
binning.py
==========
Tento modul obsahuje funkce pro binning (kategorizaci) spojitých hodnot proxy metrik.
"""
from __future__ import annotations
from typing import Optional


def bin_altitude(hf: float, quantile: float) -> str:
    """
    Kategorizuje výšku na 'low' nebo 'high' na základě zadaného kvantilu HF energie.
    """
    if hf >= quantile:
        return "low"  # hodně detailů -> nízko
    return "high"  # málo detailů -> vysoko


def bin_view(ent: float, t: Optional[float] = None) -> str:
    """
    Kategorizuje úhel pohledu na 'nadir' nebo 'oblique'.
    """
    if t is None:
        t = 1.8
    if ent >= t:
        return "nadir"
    return "oblique"


def bin_cover(ratio: float, threshold: float = 0.5) -> str:
    """
    Kategorizuje pokrytí na 'dense' nebo 'sparse'.
    """
    if ratio >= threshold:
        return "dense"
    return "sparse"


def bin_lighting(mean_int: float, threshold: Optional[float] = None) -> str:
    """
    Kategorizuje osvětlení na 'dark' nebo 'bright'.
    """
    if threshold is None:
        threshold = 115
    if mean_int < threshold:
        return "dark"
    return "bright"
