"""
stratification.py
=================
Modul pro stratifikaci snímků na základě agro-proxy metrik a cílových limitů.

Breaking change: explicitní `targets` (kombinované klíče) byly odstraněny.
Konfigurace nyní musí používat `targets_axes` (per-axis targets).
"""

from typing import List, Dict, Optional
from collections import defaultdict
from itertools import product
from bfe.frame_info import FrameInfo
import logging

logger = logging.getLogger(__name__)


class AgroStratifier:
    """
    Převádí kontinuální proxy -> binned straty a udržuje výběr podle YAML targetů.

    Chování:
      - Očekává `stratification.targets_axes` jako mapu axis -> {value: weight}.
      - Pokud `targets_axes` chybí, použije se uniformní rozdělení po osách.
      - Vytvoří interní `self.targets` mapu (combo_key -> normalized ratio, suma==1.0).
    """

    def __init__(self, config: Dict):
        sconf = (config or {}).get("stratification", {})
        self.axes = sconf.get(
            "axes",
            {
                "altitude": ["low", "high"],
                "view": ["nadir", "oblique"],
                "cover": ["sparse", "dense"],
                "lighting": ["dark", "bright"],
            },
        )

        # Per-axis targets (breaking change)
        self.targets_axes_raw: Dict[str, Dict[str, float]] = sconf.get(
            "targets_axes", {}
        )

        # Limits / protections (unchanged)
        self.limits = sconf.get(
            "limits", {"windy_max_ratio": 0.15, "sparse_max_ratio": 0.30}
        )

        self.counts: Dict[str, int] = defaultdict(int)
        self.total_selected: int = 0

        # All possible combo keys (cartesian product)
        self.combinations = self._all_combinations()

        # Compile per-combination normalized targets (sums to 1.0)
        self.targets = self._compile_targets()

    def _all_combinations(self) -> List[str]:
        """
        Vytvoří všechny možné kombinace stratifikace.
        """
        keys = list(self.axes.keys())
        values = [self.axes[k] for k in keys]
        combos = []
        for vals in product(*values):
            parts = [f"{k}:{v}" for k, v in zip(keys, vals)]
            combos.append("|".join(parts))
        return combos

    def _normalize_axis_weights(
        self, axis_weights: Dict[str, float]
    ) -> Dict[str, float]:
        s = sum(axis_weights.values()) + 1e-12
        if s == 0:
            n = len(axis_weights) or 1
            return {k: 1.0 / n for k in axis_weights}
        return {k: float(v) / s for k, v in axis_weights.items()}

    def _compile_targets(self) -> Dict[str, float]:
        """
        Sestaví `self.targets` — map combo_key -> normalized ratio (suma == 1.0).

        Postup:
          1) Použije `targets_axes` pokud je zadané (částečné hodnoty doplní uniformně).
          2) Pokud není, použije uniformní rozdělení po osách.
          3) Vypočítá produktní skóre pro každou kombinaci a normalizuje.
        """
        axis_weights_raw: Dict[str, Dict[str, float]] = {}

        if self.targets_axes_raw:
            for axis, vals in self.axes.items():
                requested = self.targets_axes_raw.get(axis, {})
                weights = {v: float(requested.get(v, 1.0)) for v in vals}
                axis_weights_raw[axis] = self._normalize_axis_weights(weights)
        else:
            for axis, vals in self.axes.items():
                axis_weights_raw[axis] = {v: 1.0 / max(1, len(vals)) for v in vals}

        # ensure coverage & normalize
        for axis, vals in self.axes.items():
            if axis not in axis_weights_raw:
                axis_weights_raw[axis] = {v: 1.0 / max(1, len(vals)) for v in vals}
            else:
                for v in vals:
                    axis_weights_raw[axis].setdefault(v, 1.0)
                axis_weights_raw[axis] = self._normalize_axis_weights(
                    axis_weights_raw[axis]
                )

        # compute product for each combo
        combo_scores: Dict[str, float] = {}
        for combo in self.combinations:
            parts = combo.split("|")
            prod = 1.0
            for part in parts:
                k, v = part.split(":", 1)
                prod *= axis_weights_raw.get(k, {}).get(v, 0.0)
            combo_scores[combo] = prod

        # normalize combo scores to sum to 1.0
        total = sum(combo_scores.values()) + 1e-12
        if total == 0:
            n = len(combo_scores) or 1
            return {c: 1.0 / n for c in combo_scores}
        targets = {c: float(s) / total for c, s in combo_scores.items()}
        return targets

    @staticmethod
    def combo_key(altitude: str, view: str, cover: str, lighting: str) -> str:
        return f"altitude:{altitude}|view:{view}|cover:{cover}|lighting:{lighting}"

    def select(
        self,
        frames_sorted: List[FrameInfo],
        target_size: Optional[int],
        config: Optional[Dict] = None,
    ) -> List[FrameInfo]:
        """
        Vybere stratifikované snímky na základě cílů a limitů.
        """
        selection_config = (config or {}).get("selection", {})
        high_quality_threshold = selection_config.get("high_quality_threshold", 0.95)

        selected: List[FrameInfo] = []
        for f in frames_sorted:
            if target_size is not None and len(selected) >= target_size:
                break

            a, v, c, l = f.strata  # type: ignore
            key = self.combo_key(a, v, c, l)

            if c == "sparse":
                if self._current_ratio(selected, cover="sparse") >= self.limits.get(
                    "sparse_max_ratio", 1.0
                ):
                    continue

            curr_ratio = self._current_ratio(selected, key=key)
            target_ratio = self.targets.get(key, 0.0)
            if curr_ratio < target_ratio or f.ml_score > high_quality_threshold:
                selected.append(f)
                self.counts[key] += 1
                self.total_selected += 1

        return selected

    def _current_ratio(
        self,
        selected: List[FrameInfo],
        key: Optional[str] = None,
        cover: Optional[str] = None,
    ) -> float:
        if not selected:
            return 0.0
        if cover is not None:
            c = sum(1 for f in selected if f.strata and f.strata[2] == cover)
            return c / len(selected)
        if key is not None:
            c = sum(
                1 for f in selected if f.strata and self.combo_key(*f.strata) == key
            )
            return c / len(selected)
        return 0.0
