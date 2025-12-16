import yaml
import argparse
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_args_and_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """
    Merge command line arguments with configuration dictionary.
    Configuration values override default args, but explicit CLI args should override config.
    (Simple implementation: add config keys to args namespace if not present or overwrite)
    
    Args:
        args (argparse.Namespace): Command line arguments.
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        argparse.Namespace: Updated arguments.
    """
    # Flatten config or just iterate top-level keys
    for key, value in config.items():
        if isinstance(value, dict):
             # For nested dicts (like 'vqgan', 'data'), we can either flatten them 
             # or keep them as dicts in the args.
             # Let's keep them as dicts but also allow flattening for specific known params if needed.
             setattr(args, key, value)
        else:
            setattr(args, key, value)
            
    return args
