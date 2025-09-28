from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from bfe.pipeline import run_curation_pipeline, prefilter_and_process_frames, assign_strata_to_frames, select_and_deduplicate
from bfe.frame_info import FrameInfo

@patch('bfe.pipeline.prefilter_and_process_frames')
@patch('bfe.pipeline.MLFrameScorer')
@patch('bfe.pipeline.assign_strata_to_frames')
@patch('bfe.pipeline.select_and_deduplicate')
@patch('bfe.pipeline.save_manifest_and_frames')
def test_run_curation_pipeline_runs(
    mock_save_results,
    mock_select_and_deduplicate,
    mock_assign_strata_to_frames,
    mock_ml_scorer,
    mock_prefilter,
):
    """Test that the main pipeline function runs without crashing."""
    mock_prefilter.return_value = [MagicMock()]
    mock_select_and_deduplicate.return_value = [MagicMock()]

    run_curation_pipeline(
        video_path="dummy.mp4",
        out_dir="dummy_out",
        config={},
        run_params={},
    )

    mock_prefilter.assert_called_once()
    mock_ml_scorer.return_value.score.assert_called_once()
    mock_assign_strata_to_frames.assert_called_once()
    mock_select_and_deduplicate.assert_called_once()
    mock_save_results.assert_called_once()

@patch('bfe.pipeline.iter_video_frames')
def test_prefilter_and_process_frames(mock_iter_video_frames):
    """Test the pre-filtering and processing of frames."""
    # Create a dummy good frame and a bad frame
    good_frame = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    bad_frame = np.zeros((100, 100, 3), dtype=np.uint8) # This will have 0 sharpness and contrast

    mock_iter_video_frames.return_value = [
        (0, 0.0, good_frame),
        (1, 0.03, bad_frame),
    ]

    candidates = prefilter_and_process_frames(
        video_path="dummy.mp4",
        min_sharpness=5.0, # Low threshold to ensure the good frame passes
        min_contrast=5.0
    )

    assert len(candidates) == 1
    assert isinstance(candidates[0], FrameInfo)
    assert candidates[0].sharpness > 5.0
    assert candidates[0].contrast > 5.0

def test_assign_strata_to_frames():
    """Test the assignment of strata to frames."""
    frames = [
        FrameInfo(idx=0, t_sec=0.0, bgr=None, gray=None, sharpness=100, contrast=50, exposure_score=0.8, noise_score=0.9, hsv_hist=None, lowres_vec=None, embed=None, hf_energy=10, view_entropy_val=1.0, green_cover=0.2, lighting_mean=100),
        FrameInfo(idx=1, t_sec=0.1, bgr=None, gray=None, sharpness=100, contrast=50, exposure_score=0.8, noise_score=0.9, hsv_hist=None, lowres_vec=None, embed=None, hf_energy=30, view_entropy_val=2.0, green_cover=0.8, lighting_mean=200),
    ]

    assign_strata_to_frames(frames)

    assert frames[0].strata == ('high', 'oblique', 'sparse', 'dark')
    assert frames[1].strata == ('low', 'nadir', 'dense', 'bright')

def test_select_and_deduplicate():
    """Test the selection and deduplication of frames."""
    frames = []
    for i in range(5):
        fi = FrameInfo(
            idx=i, t_sec=float(i), bgr=np.array([]), gray=np.array([]),
            sharpness=100, contrast=50, exposure_score=0.8, noise_score=0.9,
            hsv_hist=np.random.rand(48), lowres_vec=np.random.rand(2304), embed=np.random.rand(2352),
            ml_score=0.5 + i * 0.1, hf_energy=i*10, view_entropy_val=i, green_cover=i*0.2, lighting_mean=i*40
        )
        frames.append(fi)
    # Add a duplicate
    frames[2].embed = frames[0].embed

    assign_strata_to_frames(frames)

    # Test greedy deduplication
    selected_greedy = select_and_deduplicate(frames, target_size=4, dedup_method='greedy', config={
        "deduplication": {"cosine_threshold": 0.99}
    })
    assert len(selected_greedy) == 4

    # Test dbscan deduplication
    selected_dbscan = select_and_deduplicate(frames, target_size=5, dedup_method='dbscan', config={
        "deduplication": {"eps": 0.1}
    })
    assert len(selected_dbscan) <= 5
