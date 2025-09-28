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

def _cartesian_strata_combinations(axes: Dict[str, List[str]]) -> List[str]:
    """
    Build combo keys in the same format used in manifest ('altitude:low|view:nadir|cover:dense|lighting:bright').
    The axes dict is expected to have the keys in the order: altitude, view, cover, lighting.
    """
    import itertools

    axis_names = list(axes.keys())
    axis_values = [axes[k] for k in axis_names]
    combos = []
    for vals in itertools.product(*axis_values):
        parts = [f"{name}:{val}" for name, val in zip(axis_names, vals)]
        combos.append("|".join(parts))
    return combos


def evaluate_targets_from_config(
    strata_distribution: Dict[str, int],
    selected_count: int,
    candidates_count: int,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Compute expected counts for stratification targets using per-axis `targets_axes`.

    Behavior:
      - Uses per-axis weights (missing values default to uniform).
      - Combines per-axis weights by product to get combination ratios.
      - Allocates expected counts using largest-remainder method to ensure sum(expected)=selected_count.
    """
    if not config:
        return {"success": True, "reason": "no_config"}

    strat_cfg = (config or {}).get("stratification", {})
    axes_cfg = strat_cfg.get("axes", {})
    targets_axes_cfg = strat_cfg.get("targets_axes", {})

    if not axes_cfg:
        return {"success": True, "reason": "no_axes_defined"}

    # Build all possible combos in the same key format
    all_combos = _cartesian_strata_combinations(axes_cfg)

    # Helper: normalize axis dict
    def _normalize(d: Dict[str, float], axis_vals: List[str]) -> Dict[str, float]:
        for v in axis_vals:
            d.setdefault(v, 1.0)
        s = sum(float(x) for x in d.values()) + 1e-12
        if s == 0:
            n = len(axis_vals) or 1
            return {v: 1.0 / n for v in axis_vals}
        return {v: float(d.get(v, 0.0)) / s for v in axis_vals}

    # 1) Derive per-axis normalized weights
    per_axis_weights: Dict[str, Dict[str, float]] = {}
    for axis, vals in axes_cfg.items():
        requested = targets_axes_cfg.get(axis, {})
        per_axis_weights[axis] = _normalize(
            {k: float(v) for k, v in requested.items()}, vals
        )

    # ensure all axes present and normalized
    for axis, vals in axes_cfg.items():
        if axis not in per_axis_weights:
            per_axis_weights[axis] = {v: 1.0 / max(1, len(vals)) for v in vals}
        else:
            per_axis_weights[axis] = _normalize(per_axis_weights[axis], vals)

    # 2) compute combo ratios by product of per-axis weights
    combo_ratios: Dict[str, float] = {}
    for combo in all_combos:
        prod = 1.0
        for part in combo.split("|"):
            k, v = part.split(":", 1)
            prod *= per_axis_weights.get(k, {}).get(v, 0.0)
        combo_ratios[combo] = prod
    # normalize
    total = sum(combo_ratios.values()) + 1e-12
    if total == 0:
        n = len(combo_ratios) or 1
        combo_ratios = {c: 1.0 / n for c in combo_ratios}
    else:
        combo_ratios = {c: float(r) / total for c, r in combo_ratios.items()}

    # 3) allocate expected counts using Largest Remainder method to match selected_count
    raw_expected = {c: combo_ratios[c] * selected_count for c in combo_ratios}
    floored = {c: int(float(raw_expected[c])) for c in raw_expected}
    remainders = {c: raw_expected[c] - floored[c] for c in raw_expected}
    allocated = sum(floored.values())
    to_allocate = max(0, selected_count - allocated)
    for c, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True)[
        :to_allocate
    ]:
        floored[c] += 1

    expected_counts = {c: int(floored[c]) for c in floored}
    actual_counts = {c: int(strata_distribution.get(c, 0)) for c in all_combos}
    shortfalls = {
        c: expected_counts[c] - actual_counts[c]
        for c in all_combos
        if expected_counts[c] > actual_counts[c]
    }

    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())
    total_shortfall = sum(shortfalls.values())
    success = total_shortfall == 0

    return {
        "success": success,
        "selected_count": selected_count,
        "candidates_count": candidates_count,
        "per_axis_weights": per_axis_weights,
        "combo_ratios": combo_ratios,
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "shortfalls": shortfalls,
        "total_expected": total_expected,
        "total_actual": total_actual,
        "total_shortfall": total_shortfall,
    }
