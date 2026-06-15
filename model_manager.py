import torch
import os
from pathlib import Path
from typing import Dict, Any
from huggingface_hub import snapshot_download

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        
        self.model_configs = {
            "chest_xray_classifier": {
                "repo_id": "microsoft/BiomedVLP-CXR-BERT-specialized",
                "files": ["pytorch_model.bin", "config.json"],
                "type": "classification"
            },
            "brain_mri_segmentor": {
                "repo_id": "Project-MONAI/MONAI-extra-test-models",
                "files": ["model.pt"],
                "type": "segmentation"
            },
            "lung_ct_analyzer": {
                "repo_id": "Project-MONAI/MONAI-extra-test-models",
                "files": ["model.pt"],
                "type": "segmentation"
            },
            "chest_ct_classifier": {
                "repo_id": "microsoft/BiomedVLP-CXR-BERT-specialized",
                "files": ["pytorch_model.bin", "config.json"],
                "type": "classification"
            }
        }
    
    def download_model(self, model_name: str) -> str:
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_dir / model_name
        config = self.model_configs[model_name]
        
        if model_path.exists():
            return str(model_path)
        
        try:
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            return str(model_path)
        except Exception as e:
            raise Exception(f"Failed to download model {model_name}: {str(e)}")
    
    def load_model(self, model_name: str, force_reload: bool = False):
        if model_name in self.loaded_models and not force_reload:
            return self.loaded_models[model_name]
        
        model_path = self.download_model(model_name)
        config = self.model_configs[model_name]
        
        try:
            import torch
            from monai.networks.nets import UNet, DenseNet
            
            if config["type"] == "classification":
                model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=8)
            elif config["type"] == "segmentation":
                model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=4,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2)
                )
            
            model.load_state_dict(torch.load(model_path / "pytorch_model.bin"))
            model.eval()
            
            self.loaded_models[model_name] = model
            return model
        
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")
    
    def unload_model(self, model_name: str):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_available_models(self):
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        model_path = self.model_dir / model_name
        
        return {
            "name": model_name,
            "repo_id": config["repo_id"],
            "type": config["type"],
            "downloaded": model_path.exists(),
            "loaded": model_name in self.loaded_models,
            "path": str(model_path)
        }

class AdvancedModelManager(ModelManager):
    def __init__(self, model_dir: str = "models"):
        super().__init__(model_dir)
        self.model_versions = {}
    
    def track_model_performance(self, model_name: str, metrics: Dict[str, float]):
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        
        self.model_versions[model_name].append({
            'timestamp': '2024-01-01',
            'metrics': metrics
        })
    
    def get_best_model(self, modality: str, anatomy: str) -> str:
        modality_models = {
            'X-RAY': 'chest_xray_classifier',
            'CT': 'chest_ct_classifier', 
            'MRI': 'brain_mri_segmentor'
        }
        
        return modality_models.get(modality, 'chest_xray_classifier')