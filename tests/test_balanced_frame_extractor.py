import os
import pytest
from unittest.mock import patch, MagicMock
from bfe.pipeline import assign_strata_to_frames
from bfe.frame_info import FrameInfo


def test_strata_assignment():
    """
    Testuje přiřazení strat na základě agro-proxy.
    """
    config = {
        "stratification": {
            "thresholds": {
                "view_entropy": 1.8,
                "cover_ratio": 0.5,
                "lighting_mean": 115,
                "altitude_quantile": 0.5,
            }
        }
    }
    frames = [
        FrameInfo(
            idx=0,
            t_sec=0.0,
            bgr=None,
            gray=None,
            sharpness=100.0,
            contrast=50.0,
            exposure_score=0.8,
            noise_score=0.2,
            hsv_hist=None,
            lowres_vec=None,
            embed=None,
            hf_energy=0.6,
            view_entropy_val=2.0,
            green_cover=0.7,
            lighting_mean=120.0,
        )
    ]
    assign_strata_to_frames(frames, config=config)
    assert frames[0].strata == ("high", "nadir", "dense", "bright")


@patch("argparse.ArgumentParser.parse_args")
@patch("balanced_frame_extractor.run_curation_pipeline")
@patch("balanced_frame_extractor.load_yaml")
@patch("os.path.exists")
@patch("shutil.rmtree")
def test_main_function(
    mock_rmtree, mock_exists, mock_load_yaml, mock_run_pipeline, mock_parse_args
):
    """Test the main CLI entrypoint function."""
    # Mock args
    mock_parse_args.return_value = MagicMock(
        video="test.mp4",
        out="output",
        config=None,
        stride=1,
        target_size=None,
        min_sharpness=80.0,
        min_contrast=20.0,
        novelty_threshold=0.3,
        dedup_method="greedy",
        manifest="manifest.json",
        overwrite=True,
        debug=False,
    )
    mock_load_yaml.return_value = {}
    mock_exists.return_value = True  # Simulate output dir exists

    # Run main
    from balanced_frame_extractor import main

    main()

    # Assertions
    mock_exists.assert_any_call("output")
    mock_rmtree.assert_called_with("output")
    mock_run_pipeline.assert_called_once()
