import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "configs/clinical_default.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            "models": {
                "auto_download": True,
                "cache_models": True,
                "device": "auto"
            },
            "analysis": {
                "default_confidence_threshold": 0.75,
                "enable_segmentation": True,
                "enable_clinical_correlation": True,
                "enable_uncertainty_quantification": True
            },
            "clinical": {
                "report_template": "radiology_standard",
                "risk_stratification": True,
                "evidence_based": True
            },
            "dicom": {
                "auto_window_level": True,
                "artifact_correction": True,
                "normalization_method": "zscore"
            },
            "ui": {
                "theme": "light",
                "show_confidence_scores": True,
                "auto_generate_reports": True
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "configs/clinical_default.yaml"):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_analysis_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("analysis", {})

def get_clinical_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("clinical", {})

def update_config(section: str, key: str, value: Any):
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_default_analysis_params() -> Dict[str, Any]:
    config = load_config()
    analysis = config.get("analysis", {})
    
    return {
        "confidence_threshold": analysis.get("default_confidence_threshold", 0.75),
        "enable_segmentation": analysis.get("enable_segmentation", True),
        "enable_clinical_correlation": analysis.get("enable_clinical_correlation", True)
    }