#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI wrapper — Agro Frame Curation
=====================================
Endpoints:
  - GET  /health                      : liveness probe
  - GET  /config                      : return loaded YAML config
  - POST /curate                      : upload a video, run agro curation, return manifest + paths
  - GET  /download                     : (optional) serve a zipped result directory by absolute path

Run (dev):
    pip install fastapi uvicorn pyyaml opencv-python numpy scikit-learn
    uvicorn fastapi_app_agro:app --reload --host 0.0.0.0 --port 8000

Config:
- Reads path from env CURATION_CONFIG (default: curation_config.agro.yaml).
- Request form-fields can override YAML (stride, target_size, min_sharpness, min_contrast, manifest_name).
- Uses ml_curation_agro.py in the same directory or on PYTHONPATH.
"""
from __future__ import annotations

import os
import io
import shutil
import tempfile
import zipfile
from typing import Optional, Dict, Any

import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

# Import the agro curation pipeline
import ml_curation_agro as cur

CURATION_CONFIG = os.environ.get("CURATION_CONFIG", "curation_config.agro.yaml")

app = FastAPI(title="Agro Frame Curation", version="0.2.0")


# -----------------------------
# Config
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

CONFIG = load_config(CURATION_CONFIG)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {"config_path": os.path.abspath(CURATION_CONFIG), "config": CONFIG}


# -----------------------------
# Helpers
# -----------------------------
def _zip_dir(src_dir: str, zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                abspath = os.path.join(root, fn)
                rel = os.path.relpath(abspath, start=src_dir)
                zf.write(abspath, rel)
    return zip_path


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/curate")
async def curate_endpoint(
    file: UploadFile = File(..., description="Video file (mp4/mov/mkv/avi/m4v)"),
    # Optional overrides (fallback to YAML)
    stride: Optional[int] = Form(None),
    target_size: Optional[int] = Form(None),
    min_sharpness: Optional[float] = Form(None),
    min_contrast: Optional[float] = Form(None),
    manifest_name: Optional[str] = Form(None),
    return_zip: Optional[bool] = Form(False),
) -> JSONResponse:
    # check extension best-effort
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".m4v")):
        raise HTTPException(status_code=400, detail="Unsupported file extension. Upload a video (mp4/mov/mkv/avi/m4v).")

    # temp working dir
    workdir = tempfile.mkdtemp(prefix="agro_curation_")
    video_path = os.path.join(workdir, file.filename)
    out_dir = os.path.join(workdir, "out")

    try:
        # save uploaded file
        with open(video_path, "wb") as f:
            data = await file.read()
            f.write(data)

        # resolve config
        conf = CONFIG.get("curation", {})
        stride_v = int(stride if stride is not None else conf.get("stride", 2))
        target_size_v = int(target_size if target_size is not None else conf.get("target_size", 500))
        min_sharpness_v = float(min_sharpness if min_sharpness is not None else conf.get("min_sharpness", 80.0))
        min_contrast_v = float(min_contrast if min_contrast is not None else conf.get("min_contrast", 20.0))
        manifest_v = str(manifest_name if manifest_name is not None else conf.get("manifest_name", "manifest.json"))

        # run curation
        cfg_full = CONFIG  # pass whole YAML to enable stratification targets/limits
        _ = cur.curate_video(
            video_path=video_path,
            out_dir=out_dir,
            stride=stride_v,
            target_size=target_size_v,
            min_sharpness=min_sharpness_v,
            min_contrast=min_contrast_v,
            manifest_name=manifest_v,
            config=cfg_full
        )

        # load manifest.json
        manifest_path = os.path.join(out_dir, manifest_v)
        manifest = {}
        if os.path.exists(manifest_path):
            import json
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)

        payload = {
            "workdir": workdir,
            "out_dir": out_dir,
            "manifest_path": manifest_path,
            "params": {
                "stride": stride_v,
                "target_size": target_size_v,
                "min_sharpness": min_sharpness_v,
                "min_contrast": min_contrast_v,
                "manifest_name": manifest_v
            },
            "manifest": manifest
        }

        if return_zip:
            zip_path = os.path.join(workdir, "curated_out.zip")
            _zip_dir(out_dir, zip_path)
            payload["zip_path"] = zip_path

        return JSONResponse(payload)

    except Exception as e:
        # leave workdir for inspection; in prod you may want to clean it
        raise

# Optional: simple file server for the created ZIP (use with caution in production)
@app.get("/download")
def download(path: str = Query(..., description="Absolute path to a file to serve (e.g., ZIP created by /curate)")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=os.path.basename(path))
