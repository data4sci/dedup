import asyncio
import functools
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bfe.pipeline import run_curation_pipeline
from bfe.utils import load_yaml, setup_logging

# --- Globals & Configuration ---
CONFIG_PATH = "config.yaml"
CONFIG = load_yaml(CONFIG_PATH)
DEFAULTS = CONFIG.get("defaults", {})
PROJECT_ROOT = Path.cwd().resolve()

# --- FastAPI App Initialization ---
app = FastAPI(title="Balanced Frame Extractor", version="1.0.0")

# --- Templates and Static Files ---
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Logging ---
logger = logging.getLogger(__name__)
setup_logging(log_dir=".", debug=True)


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    context = {"request": request, "defaults": DEFAULTS}
    return templates.TemplateResponse("index.html", context)


@app.post("/process", response_class=JSONResponse)
async def process_video(
    request: Request,
    video_file: UploadFile = File(...),
    out: str = Form(None),
    overwrite: bool = Form(False),
    stride: int = Form(DEFAULTS.get("stride", 1)),
    target_size: int = Form(None),
    min_sharpness: float = Form(DEFAULTS.get("min_sharpness", 80.0)),
    min_contrast: float = Form(DEFAULTS.get("min_contrast", 20.0)),
    novelty_threshold: float = Form(DEFAULTS.get("novelty_threshold", 0.3)),
    dedup_method: str = Form(DEFAULTS.get("dedup_method", "greedy")),
):
    logger.info(
        f"Uploaded filename for default dir generation: '{video_file.filename}'"
    )
    if out:
        output_dir = Path(out)
    else:
        video_stem = Path(video_file.filename).stem
        output_dir = Path("data") / "output" / video_stem

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not overwrite:
        raise HTTPException(
            status_code=400,
            detail=f"Output directory '{output_dir}' already exists. Use 'overwrite' to replace it.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / video_file.filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    start_time = time.time()

    run_params = {
        "video": {"value": video_file.filename, "source": "web"},
        "out": {"value": str(output_dir), "source": "web" if out else "derived"},
        "overwrite": {"value": overwrite, "source": "web"},
        "stride": {"value": stride, "source": "web"},
        "target_size": {"value": target_size, "source": "web"},
        "min_sharpness": {"value": min_sharpness, "source": "web"},
        "min_contrast": {"value": min_contrast, "source": "web"},
        "novelty_threshold": {"value": novelty_threshold, "source": "web"},
        "dedup_method": {"value": dedup_method, "source": "web"},
    }

    loop = asyncio.get_event_loop()
    run_pipeline_partial = functools.partial(
        run_curation_pipeline,
        video_path=video_path,
        out_dir=output_dir,
        config=CONFIG,
        stride=stride,
        target_size=target_size,
        min_sharpness=min_sharpness,
        min_contrast=min_contrast,
        novelty_threshold=novelty_threshold,
        dedup_method=dedup_method,
        manifest_name="manifest.json",
        run_params=run_params,
    )
    await loop.run_in_executor(None, run_pipeline_partial)

    elapsed_time = time.time() - start_time

    # Update manifest with elapsed time
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r+") as f:
            manifest_data = json.load(f)
            manifest_data["task_summary"]["elapsed_time_sec"] = elapsed_time
            f.seek(0)
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
            f.truncate()

    try:
        job_path = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        job_path = output_dir

    return JSONResponse({"success": True, "results_path": f"/results?job={job_path}"})


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    requested_path = (PROJECT_ROOT / file_path).resolve()
    if not requested_path.is_relative_to(PROJECT_ROOT):
        raise HTTPException(
            status_code=403, detail="File path is outside of project directory."
        )
    if not requested_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(requested_path)


@app.get("/results", response_class=HTMLResponse)
async def get_results(request: Request, job: str):
    output_dir = (PROJECT_ROOT / job).resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    axes = list(manifest.get("task_summary", {}).get("axes_summary", {}).keys())
    for frame in manifest.get("frames", []):
        saved_path = Path(frame["saved_path"])
        try:
            relative_path = saved_path.relative_to(PROJECT_ROOT)
            frame["web_path"] = f"/files/{relative_path}"
        except ValueError:
            frame["web_path"] = ""

        if frame.get("strata"):
            frame["strata_combined"] = [
                f"{axis}:{stratum}" for axis, stratum in zip(axes, frame["strata"])
            ]

    run_params = manifest.get("run_params", {})
    task_summary = manifest.get("task_summary", {})
    targets_eval = manifest.get("targets_evaluation", {})
    elapsed_time = task_summary.get("elapsed_time_sec")

    strat_table = []
    if targets_eval and targets_eval.get("success") is not None:
        for axis, weights in targets_eval.get("per_axis_weights", {}).items():
            for value, target_pct in weights.items():
                actual_count = sum(
                    c
                    for k, c in targets_eval.get("actual_counts", {}).items()
                    if f"{axis}:{value}" in k
                )
                actual_pct = (
                    actual_count / max(task_summary.get("selected_count", 1), 1)
                ) * 100
                satisfied = actual_pct >= (target_pct * 100)
                strat_table.append(
                    {
                        "axis": axis,
                        "value": value,
                        "actual": f"{actual_pct:.1f}",
                        "target": f"{target_pct * 100:.1f}",
                        "satisfied": "✅" if satisfied else "❌",
                    }
                )

    warnings = []
    recommendations = []
    target_size = run_params.get("target_size", {}).get("value")
    if target_size and task_summary.get("selected_count", 0) < target_size:
        warnings.append(
            f"Target size not reached ({task_summary.get('selected_count', 0)}/{target_size} frames)."
        )
        recommendations.extend(
            [
                "Loosen pre-filtering: reduce --min-sharpness or --min-contrast to get more candidates.",
                "Sample more densely: reduce --stride to process more frames from the video.",
                "Adjust novelty scoring: lower --novelty-threshold to be less strict about what is a 'new' frame.",
                "Relax deduplication: lower the 'cosine_threshold' in your config to keep more unique frames.",
            ]
        )

    if targets_eval and not targets_eval.get("success"):
        warnings.append("Stratification targets were not met.")
        if not any("Loosen pre-filtering" in r for r in recommendations):
            recommendations.append(
                "The pool of candidates might be too small or skewed. Consider loosening pre-filtering or sampling more densely."
            )

    context = {
        "request": request,
        "manifest": manifest,
        "output_dir": str(output_dir),
        "elapsed_time": elapsed_time,
        "run_params": run_params,
        "task_summary": task_summary,
        "strat_table": strat_table,
        "targets_eval": targets_eval,
        "warnings": warnings,
        "recommendations": recommendations,
    }
    return templates.TemplateResponse("results.html", context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
