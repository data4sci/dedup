"""
proxies.py
==========
Tento modul obsahuje funkce pro výpočet agro proxy metrik, jako je výška letu, entropie pohledu, pokrytí zelení a osvětlení.

"""

from typing import Optional, Dict
import numpy as np
import cv2


def altitude_proxy(gray: np.ndarray, config: Optional[Dict] = None) -> float:
    """
    Vypočítá proxy pro výšku letu na základě high-pass energie.
    """
    image_proc_config = (config or {}).get("image_processing", {})
    gaussian_kernel_size = tuple(image_proc_config.get("gaussian_kernel_size", [3, 3]))
    gaussian_sigma = image_proc_config.get("gaussian_sigma", 0)

    hp = gray.astype(np.float32) - cv2.GaussianBlur(
        gray, gaussian_kernel_size, gaussian_sigma
    )
    return float(np.mean(np.abs(hp)))


def view_entropy(
    gray: np.ndarray, bins: Optional[int] = None, config: Optional[Dict] = None
) -> float:
    """
    Vypočítá entropii pohledu na základě orientací gradientů.
    """
    if bins is None:
        proxies_config = (config or {}).get("proxies", {})
        bins = proxies_config.get("view_entropy_bins", 8)

    image_proc_config = (config or {}).get("image_processing", {})
    constants_config = (config or {}).get("constants", {})

    sobel_kernel_size = image_proc_config.get("sobel_kernel_size", 3)
    epsilon = constants_config.get("epsilon", 1e-9)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    hist, _ = np.histogram(
        ang.ravel(), bins=bins, range=(0, 2 * np.pi), weights=mag.ravel()
    )
    p = hist / (hist.sum() + epsilon)
    ent = -np.sum(p * np.log(p + epsilon))
    return float(ent)


def green_cover_ratio(
    bgr: np.ndarray, threshold: Optional[float] = None, config: Optional[Dict] = None
) -> float:
    """
    Vypočítá proxy pro pokrytí zelení na základě Excess Green (ExG).
    """
    if threshold is None:
        threshold = 0.6
    small_eps = (config or {}).get("constants", {}).get("epsilon_small", 1e-6)
    b, g, r = cv2.split(bgr.astype(np.float32) + small_eps)
    exg = 2 * g - r - b
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-9)
    return float(np.mean(exg_norm >= threshold))


def classify_lighting(gray: np.ndarray) -> float:
    """
    Klasifikuje osvětlení na základě průměrné intenzity.
    """
    return float(np.mean(gray))
