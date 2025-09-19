#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rychlý dedup snímků z videa
---------------------------
Greedy "online" deduplikace ostrých snímků s automatickým prahem odvozeným
z distribuce podobností sousedních snímků na začátku videa (kalibrační okno).
Reprezentace: LowRes embedding (normalizovaný) + cosine similarity.

Navrženo tak, aby šlo snadno "zabalit" do FastAPI (funkce jsou čisté a znovupoužitelné).
Zároveň k dispozici CLI rozhraní pro dávkové použití.

Závislosti: opencv-python, numpy
    pip install opencv-python numpy

Autor: ChatGPT
Licence: MIT
"""
from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import cv2


# -----------------------------
# Pomocné datové struktury
# -----------------------------
@dataclass
class FrameInfo:
    """
    Metadata o snímku potřebná pro výběr a uložení.

    Attributes:
        index: Index snímku ve videu (0-based)
        t_sec: Timestamp snímku v sekundách
        sharpness: Ostrost snímku (variance of Laplacian)
        embedding: Normalizovaný vektor reprezentace (LowRes)
        bgr: Syrový obrázek v BGR (může být None u kalibračních snímků)
    """

    index: int
    t_sec: float
    sharpness: float
    embedding: np.ndarray
    bgr: Optional[np.ndarray]

    def __lt__(self, other: "FrameInfo") -> bool:
        """Porovnání podle ostrosti - pro pohodlné max(...) podle ostrosti."""
        return self.sharpness < other.sharpness


# -----------------------------
# Výpočet metrik/reprezentace
# -----------------------------
def variance_of_laplacian(gray: np.ndarray) -> float:
    """
    Měří ostrost snímku pomocí variance Laplacianu.

    Args:
        gray: Grayscale obrázek jako numpy array

    Returns:
        Ostrost jako float - vyšší hodnota = ostřejší snímek

    Note:
        Rychlá ale hrubá metrika. Pro přesnější měření by šlo použít
        gradient-based metriky nebo FFT-based přístupy.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def lowres_embedding(
    bgr: np.ndarray, size: Tuple[int, int] = (64, 36), to_gray: bool = True
) -> np.ndarray:
    """
    Vytvoří LowRes embedding z obrázku.

    Proces:
    1. Zmenší obrázek na zadanou velikost (rychlá feature extrakce)
    2. Volitelně převede na grayscale (menší rozměrnost)
    3. Standardizuje (odečte průměr)
    4. L2-normalizuje na jednotkovou délku

    Args:
        bgr: Vstupní obrázek v BGR formátu
        size: Cílová velikost jako (width, height)
        to_gray: Pokud True, převede na grayscale před flatten

    Returns:
        L2-normalizovaný vektor jako np.float32

    Note:
        Pro rychlost používáme INTER_AREA - nejlepší pro downsampling.
        Standardizace + normalizace činí embedding robustnější vůči
        globálním změnám osvětlení.
    """
    w, h = size[0], size[1]
    # Pozor: OpenCV resize používá (width, height) pořadí
    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

    if to_gray:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Flatten do 1D vektoru
    vec = small.astype(np.float32).reshape(-1)

    # Standardizace (zero-mean) - odolnost vůči globálním změnám jasu
    vec = vec - vec.mean()

    # L2 normalizace na jednotkovou délku
    norm = np.linalg.norm(vec) + 1e-9  # epsilon pro stabilitu
    return (vec / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Vypočítá cosine similarity mezi dvěma vektory.

    Pro L2-normalizované vektory je cosine similarity jen skalární součin.
    Pro jistotu normalizujeme znovu (kdyby přišla nenormalizovaná data).

    Args:
        a, b: Vektory pro porovnání

    Returns:
        Cosine similarity v rozsahu [-1, 1], typicky [0, 1] pro pozitivní data

    Note:
        Vyšší hodnota = podobnější vektory
        Pro normalizovaná data: 1.0 = identické, 0.0 = ortogonální
    """
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


# -----------------------------
# Čtení videa a iterace snímků
# -----------------------------
def iter_video_frames(
    video_path: str,
    stride: int = 1,
    min_sharpness: float = 0.0,
    embed_size: Tuple[int, int] = (64, 36),
    to_gray: bool = True,
    keep_images: bool = True,
) -> Iterable[FrameInfo]:
    """
    Iteruje video po snímcích s preprocessing filtry.

    Proces:
    1. Otevře video soubor
    2. Projde snímky s krokem `stride`
    3. Filtruje neostré snímky (< min_sharpness)
    4. Pro každý vyhovující snímek vytvoří embedding
    5. Vrací FrameInfo objekty

    Args:
        video_path: Cesta k video souboru
        stride: Krok pro výběr snímků (1 = každý snímek)
        min_sharpness: Minimální požadovaná ostrost
        embed_size: Velikost pro LowRes embedding
        to_gray: Použít grayscale v embeddingu
        keep_images: Uložit raw BGR data do FrameInfo

    Yields:
        FrameInfo objekty pro vyhovující snímky

    Raises:
        RuntimeError: Pokud nelze otevřít video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít video: {video_path}")

    # Získání FPS pro výpočet timestamps
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Aplikuj stride (např. každý 2. snímek)
        if idx % stride != 0:
            idx += 1
            continue

        # Výpočet timestampu
        t_sec = idx / fps

        # Výpočet ostrosti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(gray)

        # Filtruj neostré snímky
        if sharp < min_sharpness:
            idx += 1
            continue

        # Vytvoř embedding
        emb = lowres_embedding(frame, size=embed_size, to_gray=to_gray)

        # Vytvoř FrameInfo objekt
        fi = FrameInfo(
            index=idx,
            t_sec=t_sec,
            sharpness=sharp,
            embedding=emb,
            bgr=(frame.copy() if keep_images else None),  # copy pro bezpečnost
        )
        yield fi
        idx += 1

    cap.release()


# -----------------------------
# Kalibrace prahu ze začátku videa
# -----------------------------
def calibrate_threshold(
    video_path: str,
    calib_seconds: float = 15.0,
    stride: int = 1,
    min_sharpness: float = 0.0,
    embed_size: Tuple[int, int] = (64, 36),
    to_gray: bool = True,
    quantile: float = 0.95,
    min_fallback: float = 0.985,
    debug: bool = False,
) -> Tuple[float, dict]:
    """
    Kalibruje práh podobnosti z distribuce sousedních snímků.

    Spočte distribuci podobností mezi sousedními snímky v prvních N sekundách
    videa a odvozuje práh jako kvantil této distribuce.

    Args:
        video_path: Cesta k video souboru
        calib_seconds: Délka kalibračního okna v sekundách
        stride: Krok pro výběr snímků
        min_sharpness: Minimální ostrost pro zařazení do kalibrace
        embed_size: Velikost LowRes embeddingu
        to_gray: Použít grayscale v embeddingu
        quantile: Kvantil pro odvození prahu (vyšší = přísnější)
        min_fallback: Minimální fallback práh při selhání kalibrace
        debug: Vypsat debug informace

    Returns:
        Tuple[práh_podobnosti, debug_statistiky]

    Note:
        Vyšší kvantil znamená přísnější práh (méně snímků se uloží).
        Pro quantile >= 0.98 se automaticky aplikuje min_fallback.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_index = int(calib_seconds * fps)

    if debug:
        print(f"[DEBUG] Kalibrace na prvních {calib_seconds}s (≈{max_index} snímků)")

    # Sběr podobností mezi sousedními snímky
    idx = 0
    sims: List[float] = []
    prev_emb: Optional[np.ndarray] = None
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or idx > max_index:
            break

        if idx % stride != 0:
            idx += 1
            continue

        # Filtruj neostré snímky
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(gray)
        if sharp < min_sharpness:
            idx += 1
            continue

        # Vytvoř embedding a porovnej s předchozím
        emb = lowres_embedding(frame, size=embed_size, to_gray=to_gray)
        if prev_emb is not None:
            sim = cosine_similarity(prev_emb, emb)
            sims.append(sim)

        prev_emb = emb
        processed_frames += 1
        idx += 1

    cap.release()

    # Výpočet statistik
    if not sims:
        if debug:
            print(
                f"[WARNING] Žádné podobnosti pro kalibraci! Použit fallback: {min_fallback}"
            )
        return min_fallback, {"samples": 0, "fallback_used": True}

    sims_array = np.array(sims, dtype=np.float32)
    stats = {
        "samples": len(sims),
        "processed_frames": processed_frames,
        "min_sim": float(sims_array.min()),
        "max_sim": float(sims_array.max()),
        "mean_sim": float(sims_array.mean()),
        "median_sim": float(np.median(sims_array)),
        "fallback_used": False,
    }

    # Výpočet prahu jako kvantil
    q = float(np.quantile(sims_array, quantile))

    # Agresivnější omezení a speciální handling pro vysoké kvantily
    if quantile >= 0.98:
        # Pro velmi vysoké kvantily buď extra opatrný
        result = max(max(q, min_fallback), 0.99)
    else:
        # Standardní omezení do rozumných mezí
        result = max(0.85, min(0.9999, q))

    stats["raw_quantile"] = q
    stats["final_threshold"] = result

    if debug:
        print(f"[DEBUG] Kalibrační statistiky:")
        print(f"  - Zpracováno snímků: {processed_frames}")
        print(f"  - Podobností pro analýzu: {len(sims)}")
        print(
            f"  - Min/Med/Max podobnost: {stats['min_sim']:.4f}/{stats['median_sim']:.4f}/{stats['max_sim']:.4f}"
        )
        print(f"  - Raw Q{quantile:.2f}: {q:.4f}")
        print(f"  - Finální práh: {result:.4f}")

    return result, stats


# -----------------------------
# Greedy dedup s force-pick mechanismem
# -----------------------------
def greedy_dedup(
    video_path: str,
    out_dir: str,
    stride: int = 1,
    min_sharpness: float = 0.0,
    embed_size: Tuple[int, int] = (64, 36),
    to_gray: bool = True,
    calib_seconds: float = 15.0,
    quantile: float = 0.95,
    force_every_sec: float = 3.0,
    jpeg_quality: int = 95,
    respect_threshold_on_force: bool = True,
    disable_force_pick: bool = False,
    cooldown_frames: int = 0,
    debug: bool = False,
) -> List[FrameInfo]:
    """
    Jednoprůchodová greedy deduplikace s pokročilými mechanismy.

    Algoritmus:
    1. Kalibruje práh podobnosti z prvních N sekund videa
    2. Iteruje snímky a porovnává s aktuálním "pivotem"
    3. Snímky podobné pivotu (similarity >= práh) se vyřadí
    4. Snímky dostatečně odlišné se stanou novým pivotem
    5. Force-pick: pokud dlouho nebyl vybrán žádný snímek, vybere nejostřejší
       z posledního intervalu (volitelně respektuje práh podobnosti)

    Args:
        video_path: Cesta k video souboru
        out_dir: Výstupní složka pro uložené snímky
        stride: Krok pro výběr snímků (1 = každý snímek)
        min_sharpness: Minimální ostrost pro zařazení
        embed_size: Velikost LowRes embeddingu
        to_gray: Použít grayscale v embeddingu
        calib_seconds: Délka kalibračního okna
        quantile: Kvantil pro odvození prahu (vyšší = méně snímků)
        force_every_sec: Interval pro force-pick (0 = vypnuto)
        jpeg_quality: Kvalita JPEG výstupu (1-100)
        respect_threshold_on_force: I při force-pick respektovat práh
        disable_force_pick: Úplně vypnout force-pick mechanismus
        cooldown_frames: Počet snímků k přeskočení po každém keep
        debug: Vypsat debug informace

    Returns:
        Seznam vybraných FrameInfo objektů
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Kalibrace prahu s debug info
    thr_keep_same, calib_stats = calibrate_threshold(
        video_path=video_path,
        calib_seconds=calib_seconds,
        stride=stride,
        min_sharpness=min_sharpness,
        embed_size=embed_size,
        to_gray=to_gray,
        quantile=quantile,
        debug=debug,
    )

    print(
        f"[INFO] Kalibrovaný práh podobnosti: {thr_keep_same:.4f} (Q={quantile}, samples={calib_stats['samples']})"
    )
    if disable_force_pick:
        print("[INFO] Force-pick mechanismus VYPNUT - čistě greedy deduplikace")
    elif not respect_threshold_on_force:
        print("[WARNING] Force-pick NERESPEKTUJE práh - může vybrat i podobné snímky!")

    # 2) Inicializace stavu algoritmu
    kept: List[FrameInfo] = []
    pivot: Optional[FrameInfo] = None
    last_keep_time: float = -math.inf
    window_best: Optional[FrameInfo] = None

    # Cooldown mechanismus
    cooldown_counter: int = 0

    # Debug statistiky
    total_processed = 0
    kept_by_similarity = 0
    kept_by_force = 0
    dropped_by_similarity = 0
    dropped_by_cooldown = 0
    dropped_by_force_threshold = 0

    # 3) Hlavní iterační smyčka
    for fi in iter_video_frames(
        video_path=video_path,
        stride=stride,
        min_sharpness=min_sharpness,
        embed_size=embed_size,
        to_gray=to_gray,
        keep_images=True,
    ):
        total_processed += 1

        # COOLDOWN: přeskoč snímky těsně po posledním keep
        if cooldown_counter > 0:
            cooldown_counter -= 1
            dropped_by_cooldown += 1
            if debug and total_processed % 100 == 0:
                print(f"[DEBUG] Cooldown: přeskočen snímek {fi.index}")
            continue

        # Aktualizuj nejlepší kandidát v současném okně (pro force-pick)
        if (window_best is None) or (fi.sharpness > window_best.sharpness):
            window_best = fi

        # První snímek vždy bereme jako pivot
        if pivot is None:
            _save_frame(fi, out_dir, jpeg_quality)
            kept.append(fi)
            pivot = fi
            last_keep_time = fi.t_sec
            window_best = None
            cooldown_counter = cooldown_frames
            kept_by_similarity += 1  # technicky první snímek
            if debug:
                print(f"[DEBUG] První snímek (pivot): {fi.index} @ {fi.t_sec:.1f}s")
            continue

        # FORCE-PICK: pokud uplynulo příliš času bez výběru
        time_since_last = fi.t_sec - last_keep_time
        if (
            not disable_force_pick
            and force_every_sec > 0
            and time_since_last >= force_every_sec
            and window_best is not None
        ):

            # Zkontroluj podobnost nejlepšího kandidáta k pivotu
            force_sim = cosine_similarity(window_best.embedding, pivot.embedding)

            if not respect_threshold_on_force or force_sim < thr_keep_same:
                # Force-pick prošel - ulož nejostřejší snímek z okna
                _save_frame(window_best, out_dir, jpeg_quality)
                kept.append(window_best)
                pivot = window_best
                last_keep_time = window_best.t_sec
                kept_by_force += 1
                cooldown_counter = cooldown_frames
                if debug:
                    print(
                        f"[DEBUG] Force pick: snímek {window_best.index} @ {window_best.t_sec:.1f}s (sim={force_sim:.4f})"
                    )
            else:
                # Force-pick neprošel kvůli prahu - jen posun časovač
                last_keep_time = fi.t_sec
                dropped_by_force_threshold += 1
                if debug:
                    print(
                        f"[DEBUG] Force pick zamítnut: sim={force_sim:.4f} >= {thr_keep_same:.4f}"
                    )

            window_best = None
            continue

        # GREEDY ROZHODNUTÍ: porovnání aktuálního snímku s pivotem
        sim = cosine_similarity(fi.embedding, pivot.embedding)

        if sim >= thr_keep_same:
            # Příliš podobné -> drop
            dropped_by_similarity += 1
            if debug and total_processed % 50 == 0:
                print(
                    f"[DEBUG] Drop: snímek {fi.index} (sim={sim:.4f} >= {thr_keep_same:.4f})"
                )
            continue

        # Dostatečně odlišné -> keep jako nový pivot
        _save_frame(fi, out_dir, jpeg_quality)
        kept.append(fi)
        pivot = fi
        last_keep_time = fi.t_sec
        window_best = None
        kept_by_similarity += 1
        cooldown_counter = cooldown_frames
        if debug:
            print(
                f"[DEBUG] Keep: snímek {fi.index} @ {fi.t_sec:.1f}s (sim={sim:.4f} < {thr_keep_same:.4f})"
            )

    # 4) Závěrečné zpracování: pokud zůstal nějaký kandidát v okně
    if (
        window_best is not None
        and not disable_force_pick
        and kept
        and window_best.t_sec > kept[-1].t_sec
    ):

        final_sim = (
            cosine_similarity(window_best.embedding, pivot.embedding) if pivot else 0.0
        )
        if not respect_threshold_on_force or final_sim < thr_keep_same:
            _save_frame(window_best, out_dir, jpeg_quality)
            kept.append(window_best)
            kept_by_force += 1
            if debug:
                print(f"[DEBUG] Finální force pick: snímek {window_best.index}")

    # 5) Výsledné statistiky
    print(f"\n[INFO] === DEDUPLIKACE DOKONČENA ===")
    print(f"Celkem zpracováno snímků: {total_processed}")
    print(f"Uloženo snímků: {len(kept)}")
    print(f"  - Podle podobnosti: {kept_by_similarity}")
    print(f"  - Force pick: {kept_by_force}")
    print(f"Vyhozeno snímků: {total_processed - len(kept)}")
    print(f"  - Příliš podobné: {dropped_by_similarity}")
    print(f"  - Cooldown: {dropped_by_cooldown}")
    print(f"  - Force pick zamítnut: {dropped_by_force_threshold}")
    print(
        f"Komprese: {len(kept)}/{total_processed} = {100*len(kept)/max(total_processed,1):.1f}%"
    )
    print(f"Výstupní složka: {out_dir}")

    return kept


def _save_frame(fi: FrameInfo, out_dir: str, jpeg_quality: int = 95) -> None:
    """
    Uloží snímek na disk s informativním názvem souboru.

    Formát názvu: frame_XXXXXX_tYYYYYYY.YYY_sharpZZZZZ.Z.jpg
    - XXXXXX: 6-místný index snímku (zero-padded)
    - YYYYYYY.YYY: 10-místný timestamp (s přesností na milisekundy)
    - ZZZZZ.Z: 8-místná ostrost (s jedním desetinným místem)

    Args:
        fi: FrameInfo objekt s metadaty snímku
        out_dir: Výstupní složka
        jpeg_quality: Kvalita JPEG komprese (1-100)
    """
    if fi.bgr is None:
        return

    fname = f"frame_{fi.index:06d}_t{fi.t_sec:010.3f}_sharp{fi.sharpness:08.1f}.jpg"
    path = os.path.join(out_dir, fname)

    # Uložení s danou JPEG kvalitou
    success = cv2.imwrite(
        path, fi.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )

    if not success:
        print(f"[WARNING] Nepodařilo se uložit snímek: {path}")


# -----------------------------
# CLI rozhraní
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    """
    Vytvoří argument parser s všemi možnostmi konfigurace.
    """
    p = argparse.ArgumentParser(
        description=(
            "Greedy deduplikace snímků z videa s automaticky kalibrovaným prahem. "
            "Používá LowRes embedding + cosine similarity s pokročilými mechanismy "
            "pro zabránění výběru příliš podobných snímků."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Základní parametry
    p.add_argument("video", help="Cesta k video souboru")
    p.add_argument(
        "-o", "--out", default="out_frames", help="Výstupní složka pro uložené snímky"
    )

    # Preprocessing parametry
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Vybírat každý N-tý snímek z videa (zrychlení)",
    )
    p.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Minimální ostrost (Variance of Laplacian) pro zařazení snímku",
    )

    # Embedding parametry
    p.add_argument(
        "--embed-width", type=int, default=64, help="Šířka LowRes embeddingu"
    )
    p.add_argument(
        "--embed-height", type=int, default=36, help="Výška LowRes embeddingu"
    )
    p.add_argument(
        "--no-gray",
        action="store_true",
        help="Nepoužívat grayscale v embeddingu (ponechá BGR flatten)",
    )

    # Kalibrační parametry
    p.add_argument(
        "--calib-seconds",
        type=float,
        default=15.0,
        help="Délka kalibračního okna v sekundách",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Kvantil pro odvození prahu podobnosti (0.0-1.0, vyšší = přísnější)",
    )
    p.add_argument(
        "--min-fallback",
        type=float,
        default=0.985,
        help="Minimální fallback práh při selhání kalibrace",
    )

    # Force-pick mechanismus
    p.add_argument(
        "--force-every-sec",
        type=float,
        default=3.0,
        help="Vynutit výběr nejostřejšího snímku po N sekundách bez výběru (0 = vypnuto)",
    )
    p.add_argument(
        "--disable-force-pick",
        action="store_true",
        help="Úplně vypnout force-pick mechanismus (čistě greedy podle podobnosti)",
    )
    p.add_argument(
        "--no-respect-threshold-on-force",
        action="store_true",
        help="Force-pick ignoruje práh podobnosti (může vybrat i podobné snímky)",
    )

    # Cooldown mechanismus
    p.add_argument(
        "--cooldown-frames",
        type=int,
        default=0,
        help="Počet snímků k přeskočení po každém výběru (zabránění oscilacím)",
    )

    # Výstupní parametry
    p.add_argument(
        "--jpeg-quality", type=int, default=95, help="Kvalita JPEG výstupu (1-100)"
    )

    # Debug a diagnostika
    p.add_argument(
        "--debug",
        action="store_true",
        help="Vypsat detailní debug informace během zpracování",
    )

    return p


def main():
    """
    Hlavní funkce pro CLI použití.

    Zpracuje argumenty příkazové řádky a spustí deduplikaci
    s odpovídajícími parametry.
    """
    ap = build_argparser()
    args = ap.parse_args()

    # Validace argumentů
    if not os.path.exists(args.video):
        print(f"[ERROR] Video soubor neexistuje: {args.video}")
        return 1

    if args.quantile < 0.0 or args.quantile > 1.0:
        print(f"[ERROR] Quantile musí být v rozsahu 0.0-1.0, zadáno: {args.quantile}")
        return 1

    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        print(
            f"[ERROR] JPEG kvalita musí být v rozsahu 1-100, zadáno: {args.jpeg_quality}"
        )
        return 1

    # Příprava parametrů
    embed_size = (int(args.embed_width), int(args.embed_height))
    to_gray = not args.no_gray
    respect_threshold_on_force = not args.no_respect_threshold_on_force

    # Informační výpis konfigurace
    print(f"[INFO] === KONFIGURACE DEDUPLIKACE ===")
    print(f"Video: {args.video}")
    print(f"Výstup: {args.out}")
    print(f"Stride: {args.stride}")
    print(f"Min. ostrost: {args.min_sharpness}")
    print(
        f"Embedding: {embed_size[0]}x{embed_size[1]} {'(grayscale)' if to_gray else '(BGR)'}"
    )
    print(f"Kalibrace: {args.calib_seconds}s, quantile={args.quantile}")
    print(
        f"Force pick: {'VYPNUT' if args.disable_force_pick else f'každých {args.force_every_sec}s'}"
    )
    if not args.disable_force_pick:
        print(f"  - Respektuje práh: {'ANO' if respect_threshold_on_force else 'NE'}")
    if args.cooldown_frames > 0:
        print(f"Cooldown: {args.cooldown_frames} snímků")
    print(f"JPEG kvalita: {args.jpeg_quality}%")
    print(f"Debug: {'ANO' if args.debug else 'NE'}")
    print()

    try:
        # Spuštění deduplikace
        kept_frames = greedy_dedup(
            video_path=args.video,
            out_dir=args.out,
            stride=args.stride,
            min_sharpness=args.min_sharpness,
            embed_size=embed_size,
            to_gray=to_gray,
            calib_seconds=args.calib_seconds,
            quantile=args.quantile,
            force_every_sec=args.force_every_sec,
            jpeg_quality=args.jpeg_quality,
            respect_threshold_on_force=respect_threshold_on_force,
            disable_force_pick=args.disable_force_pick,
            cooldown_frames=args.cooldown_frames,
            debug=args.debug,
        )

        print(f"\n[SUCCESS] Deduplikace úspěšně dokončena!")
        print(f"Uloženo {len(kept_frames)} snímků do {args.out}")

        return 0

    except Exception as e:
        print(f"[ERROR] Chyba během deduplikace: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
