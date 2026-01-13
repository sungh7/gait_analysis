"""
Configuration management for gait analysis.

Provides functions to load and access configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Global config instance
_config: Optional[Dict[str, Any]] = None
_config_path: Optional[str] = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file (default: config.yaml)

    Returns:
        Dictionary containing configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)

    if not path.exists():
        # Try relative to this file's directory
        alt_path = Path(__file__).parent.parent.parent / config_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get the global configuration instance.

    Uses lazy loading - config is loaded on first access.

    Args:
        reload: If True, force reload of configuration

    Returns:
        Dictionary containing configuration values
    """
    global _config

    if _config is None or reload:
        _config = load_config()

    return _config


def get_signal_processing_config(joint: str) -> Dict[str, Any]:
    """
    Get signal processing configuration for a specific joint.

    Args:
        joint: Joint name ('knee', 'hip', or 'ankle')

    Returns:
        Dictionary with joint-specific signal processing parameters
    """
    config = get_config()
    joint_lower = joint.lower()

    if joint_lower not in config.get('signal_processing', {}):
        raise ValueError(f"Unknown joint: {joint}. Valid: knee, hip, ankle")

    return config['signal_processing'][joint_lower]


def get_kinematic_constraints_config(joint: str) -> Dict[str, Any]:
    """
    Get kinematic constraints configuration for a specific joint.

    Args:
        joint: Joint name ('knee', 'hip', or 'ankle')

    Returns:
        Dictionary with joint-specific kinematic constraint parameters
    """
    config = get_config()
    joint_lower = joint.lower()

    if joint_lower not in config.get('kinematic_constraints', {}):
        raise ValueError(f"Unknown joint: {joint}. Valid: knee, hip, ankle")

    return config['kinematic_constraints'][joint_lower]


def get_path(path_name: str) -> str:
    """
    Get a configured path.

    Args:
        path_name: Name of the path (e.g., 'data_dir', 'output_dir')

    Returns:
        Path string from configuration
    """
    config = get_config()
    paths = config.get('paths', {})

    if path_name not in paths:
        raise ValueError(f"Unknown path: {path_name}")

    return paths[path_name]
