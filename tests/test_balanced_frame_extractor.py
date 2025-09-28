
import os
import pytest
from unittest.mock import patch, MagicMock

from balanced_frame_extractor import main

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
