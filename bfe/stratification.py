"""
stratification.py
=================
Tento modul obsahuje funkce pro stratifikaci snímků na základě agro proxy metrik a cílových limitů.

"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from itertools import product
from bfe.frame_info import FrameInfo


class AgroStratifier:
    """
    Převádí kontinuální proxy -> binned straty a udržuje výběr podle YAML targetů.
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
        self.targets_raw: Dict[str, float] = sconf.get("targets", {"*": 1.0})
        self.limits = sconf.get(
            "limits", {"windy_max_ratio": 0.15, "sparse_max_ratio": 0.30}
        )
        self.counts: Dict[str, int] = defaultdict(int)
        self.total_selected: int = 0

        # Expand '*' later proportionally over missing combinations
        self.combinations = self._all_combinations()

        # Compile explicit targets and distribute wildcard
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

    def _compile_targets(self) -> Dict[str, float]:
        """
        Kompiluje explicitní cíle a distribuuje wildcard hodnoty.
        """
        explicit_total = sum(v for k, v in self.targets_raw.items() if k != "*")
        wildcard = self.targets_raw.get("*", 0.0)
        remaining = max(0.0, 1.0 - explicit_total)
        if wildcard > 0:
            wildcard_share = remaining
        else:
            wildcard_share = 0.0

        explicit_keys = {k for k in self.targets_raw.keys() if k != "*"}
        missing = [c for c in self.combinations if c not in explicit_keys]
        per = (wildcard_share / len(missing)) if missing and wildcard_share > 0 else 0.0

        targets = {}
        for c in self.combinations:
            if c in self.targets_raw and c != "*":
                targets[c] = float(self.targets_raw[c])
            else:
                targets[c] = float(per)
        s = sum(targets.values()) + 1e-9
        for k in list(targets.keys()):
            targets[k] = targets[k] / s
        return targets

    @staticmethod
    def combo_key(altitude: str, view: str, cover: str, lighting: str) -> str:
        """
        Vytvoří klíč pro kombinaci stratifikace.
        """
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
        """
        Vypočítá aktuální poměr vybraných snímků pro daný klíč nebo pokrytí.
        """
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
