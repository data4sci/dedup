
import os
import yaml
import pytest
from unittest.mock import patch, MagicMock

from balanced_frame_extractor import load_yaml, run_curation_pipeline, main
import balanced_frame_extractor as bfe

def test_import():
    """Test that the main script can be imported without errors."""
    assert bfe is not None

def test_load_yaml_valid(tmp_path):
    """Test loading a valid YAML file."""
    config_data = {"defaults": {"stride": 2}}
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_yaml(str(config_file))
    assert config == config_data

def test_load_yaml_nonexistent():
    """Test loading a nonexistent YAML file."""
    config = load_yaml("nonexistent.yaml")
    assert config == {}

def test_load_yaml_invalid(tmp_path):
    """Test loading an invalid YAML file."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("key: value: another_value")
    config = load_yaml(str(config_file))
    assert config == {}

@patch('balanced_frame_extractor.prefilter_and_process_frames')
@patch('balanced_frame_extractor.MLFrameScorer')
@patch('balanced_frame_extractor.assign_strata_to_frames')
@patch('balanced_frame_extractor.select_and_deduplicate')
@patch('balanced_frame_extractor.save_results')
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


@patch('argparse.ArgumentParser.parse_args')
@patch('balanced_frame_extractor.run_curation_pipeline')
@patch('balanced_frame_extractor.load_yaml')
@patch('os.path.exists')
@patch('shutil.rmtree')
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
    mock_exists.return_value = True # Simulate output dir exists

    # Run main
    main()

    # Assertions
    mock_exists.assert_any_call("output")
    mock_rmtree.assert_called_with("output")
    mock_run_pipeline.assert_called_once()
