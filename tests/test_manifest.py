import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from bfe.frame_info import FrameInfo
from bfe.manifest import save_manifest_and_frames


@pytest.fixture
def dummy_frames():
    """Create a list of dummy FrameInfo objects for testing."""
    frame1 = FrameInfo(
        idx=0, t_sec=0.0, bgr=np.zeros((10, 10, 3), dtype=np.uint8), gray=None, sharpness=100, contrast=50, 
        exposure_score=0.8, noise_score=0.9, hsv_hist=np.array([0.1, 0.2]), lowres_vec=np.array([0.3, 0.4]), 
        embed=np.array([0.1, 0.2, 0.3, 0.4]), ml_score=0.85, subscores={"quality": 0.9, "novelty": 0.8}, 
        strata=('high', 'nadir', 'dense', 'bright'), hf_energy=20, view_entropy_val=2.0, green_cover=0.7, lighting_mean=150
    )
    return [frame1]

def test_save_manifest_and_frames(tmp_path, dummy_frames):
    """Test saving the manifest and frame images."""
    out_dir = tmp_path / "output"
    video_path = "dummy.mp4"
    num_candidates = 10
    run_params = {"stride": {"value": 1, "source": "default"}}

    save_manifest_and_frames(
        frames=dummy_frames, 
        out_dir=str(out_dir), 
        video_path=video_path, 
        num_candidates=num_candidates, 
        run_params=run_params
    )

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    
    assert manifest_data["input_video"] == os.path.abspath(video_path)
    assert manifest_data["task_summary"]["candidates_count"] == num_candidates
    assert len(manifest_data["frames"]) == 1
    assert manifest_data["frames"][0]["ml_score"] == 0.85

    # Check if the image was saved
    saved_image_path = manifest_data["frames"][0]["saved_path"]
    assert os.path.exists(saved_image_path)
