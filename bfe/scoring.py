#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoring.py
==========
Tento modul obsahuje třídu pro ML-based skórování snímků.
"""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import cv2
import logging

from bfe.frame_info import FrameInfo
from bfe.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

class MLFrameScorer:
    """
    Třída pro výpočet ML skóre snímků na základě kvality a novosti.
    """
    def __init__(
        self,
        novelty_memory: int = 64,
        novelty_threshold: float = 0.3,
        config: Optional[Dict] = None,
    ):
        self.prototypes: List[np.ndarray] = []
        self.novelty_memory = int(novelty_memory)
        self.novelty_threshold = novelty_threshold

        scoring_config = (config or {}).get("scoring", {})
        weights = scoring_config.get("weights", {})
        quality_components = scoring_config.get("quality_components", {})
        scale_ranges = scoring_config.get("scale_ranges", {})

        self.weight_quality = weights.get("quality", 0.5)
        self.weight_novelty = weights.get("content_novelty", 0.5)

        self.quality_sharpness = quality_components.get("sharpness", 0.35)
        self.quality_contrast = quality_components.get("contrast", 0.30)
        self.quality_exposure = quality_components.get("exposure", 0.25)
        self.quality_noise = quality_components.get("noise", 0.10)

        self.novelty_memory = scoring_config.get("novelty_memory", 64)
        self.sharpness_range = scale_ranges.get("sharpness", [80.0, 300.0])
        self.contrast_range = scale_ranges.get("contrast", [20.0, 80.0])

    @staticmethod
    def _scale(x: float, lo: float, hi: float) -> float:
        """Normalizuje hodnotu do rozsahu [0, 1]."""
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo + 1e-9))))

    def _quality_score(self, f: FrameInfo) -> float:
        """Vypočítá vážené skóre kvality."""
        s = self._scale(f.sharpness, self.sharpness_range[0], self.sharpness_range[1])
        c = self._scale(f.contrast, self.contrast_range[0], self.contrast_range[1])
        e = f.exposure_score
        n = f.noise_score
        return (
            self.quality_sharpness * s
            + self.quality_contrast * c
            + self.quality_exposure * e
            + self.quality_noise * n
        )

    def _novelty_score(self, emb: np.ndarray) -> float:
        """Vypočítá skóre novosti oproti nedávným prototypům."""
        if not self.prototypes:
            return 1.0
        sims = [
            cosine_similarity(emb, p) for p in self.prototypes[-self.novelty_memory :]
        ]
        max_sim = max(sims) if sims else 0.0
        return float(max(0.0, min(1.0, 1.0 - max_sim)))

    def score(self, frames: List[FrameInfo], config: Optional[Dict] = None) -> None:
        """Vypočítá a přiřadí ML skóre pro seznam snímků."""
        output_config = (config or {}).get("output", {})
        log_intervals = output_config.get("log_intervals", {})
        ml_scoring_interval = log_intervals.get("ml_scoring", 500)

        logger.debug(f"Starting ML scoring for {len(frames)} frames")
        for i, f in enumerate(frames):
            if i % ml_scoring_interval == 0 and i > 0:
                logger.debug(f"ML scoring progress: {i}/{len(frames)} frames")

            q = self._quality_score(f)
            nov = self._novelty_score(f.embed)
            total = self.weight_quality * q + self.weight_novelty * nov
            f.ml_score = float(total)
            f.subscores = {"quality": q, "content_novelty": nov}

            logger.debug(
                f"Frame {f.idx}: quality={q:.3f}, novelty={nov:.3f} → ML_score={total:.3f}"
            )

            if nov > self.novelty_threshold:
                self.prototypes.append(f.embed)
                logger.debug(f"Frame {f.idx} added to prototypes (novelty={nov:.3f})")

        logger.info(
            f"ML scoring complete. Prototypes collected: {len(self.prototypes)}"
        )
