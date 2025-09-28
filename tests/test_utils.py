import yaml
import pytest
import logging
from bfe.utils import load_yaml, setup_logging

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

def test_setup_logging(tmp_path):
    """Test the setup_logging function."""
    log_dir = tmp_path / "logs"
    setup_logging(str(log_dir), debug=True)
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) > 0
