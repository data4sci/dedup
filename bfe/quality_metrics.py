"""
quality_metrics.py
===================
Tento modul obsahuje funkce pro výpočet nízkoúrovňových metrik kvality snímků, jako je ostrost, kontrast, expozice a šum.

"""

from typing import Tuple, Optional, Dict
import numpy as np
import cv2


def variance_of_laplacian(gray: np.ndarray) -> float:
    """
    Vypočítá varianci Laplacianu pro odhad ostrosti snímku.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_contrast(gray: np.ndarray) -> float:
    """
    Odhadne kontrast snímku jako standardní odchylku intenzit.
    """
    return float(np.std(gray))


def exposure_metrics(
    gray: np.ndarray, config: Optional[Dict] = None
) -> Tuple[float, float, float]:
    """
    Vypočítá metriky expozice: průměrná intenzita, podexpozice a přeexpozice.
    """
    image_proc_config = (config or {}).get("image_processing", {})
    exposure_config = image_proc_config.get("exposure", {})
    constants_config = (config or {}).get("constants", {})

    underexposure_bins = exposure_config.get("underexposure_bins", 10)
    overexposure_start = exposure_config.get("overexposure_start", 246)
    epsilon = constants_config.get("epsilon", 1e-9)

    mean = float(np.mean(gray))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + epsilon)
    under = float(hist[:underexposure_bins].sum())
    over = float(hist[overexposure_start:].sum())
    return mean, under, over


def exposure_score_from_metrics(
    mean: float, under_frac: float, over_frac: float, config: Optional[Dict] = None
) -> float:
    """
    Vypočítá skóre expozice na základě metrik expozice.
    """
    image_proc_config = (config or {}).get("image_processing", {})
    exposure_config = image_proc_config.get("exposure", {})

    center_value = exposure_config.get("center_value", 128.0)
    center_penalty_weight = exposure_config.get("center_penalty_weight", 0.6)
    clip_penalty_weight = exposure_config.get("clip_penalty_weight", 0.4)
    clip_multiplier = exposure_config.get("clip_multiplier", 2.0)

    center_penalty = abs(mean - center_value) / center_value
    clip_penalty = clip_multiplier * (under_frac + over_frac)
    raw = 1.0 - min(
        1.0, center_penalty_weight * center_penalty + clip_penalty_weight * clip_penalty
    )
    return float(max(0.0, min(1.0, raw)))


def estimate_noise_score(gray: np.ndarray, config: Optional[Dict] = None) -> float:
    """
    Odhadne skóre šumu na základě reziduální standardní odchylky po Gaussově rozmazání.
    """
    image_proc_config = (config or {}).get("image_processing", {})

    gaussian_kernel_size = tuple(image_proc_config.get("gaussian_kernel_size", [3, 3]))
    gaussian_sigma = image_proc_config.get("gaussian_sigma", 0)
    noise_scaling_factor = image_proc_config.get("noise_scaling_factor", 25.0)

    blur = cv2.GaussianBlur(gray, gaussian_kernel_size, gaussian_sigma)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    resid_std = float(np.std(resid))
    score = 1.0 - min(1.0, resid_std / noise_scaling_factor)
    return float(max(0.0, score))
