import json
from unittest.mock import patch

from bfe.reporting import print_human_readable_statistics


def test_print_human_readable_statistics(tmp_path, capsys):
    """Test the printing of human-readable statistics."""
    manifest_data = {
        "task_summary": {
            "video": "test.mp4",
            "output_dir": "/tmp/output",
            "selected_count": 1,
            "candidates_count": 10,
            "selection_ratio": "10.0%",
        },
        "frames": [
            {"ml_score": 0.85}
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)

    print_human_readable_statistics(str(manifest_path))

    captured = capsys.readouterr()
    assert "FRAME CURATION STATISTICS" in captured.out
    assert "Video: test.mp4" in captured.out
    assert "ML SCORE" in captured.out
