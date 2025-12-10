"""Utility functions for project path management and YAML configuration loading."""
from pathlib import Path
import yaml

# Define the root directory of the project (2 levels up from this file location)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def project_path(*parts: str) -> Path:
    """Construct absolute path relative to project root.
    
    Args:
        *parts: Path components to join with PROJECT_ROOT
    
    Returns:
        Path object pointing to the specified location
    """
    return PROJECT_ROOT.joinpath(*parts)

def load_yaml(path: str):
    """Load and parse YAML configuration file from project.
    
    Args:
        path: Relative path to YAML file from project root
    
    Returns:
        Dictionary containing parsed YAML data
    """
    with open(project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
