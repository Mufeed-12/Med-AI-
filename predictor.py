import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import monai
from monai.networks.nets import UNet, DenseNet
import timm

class MedicalPredictor:
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.current_modality = None
    
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_model(self, modality: str, anatomy: str):
        model_key = f"{modality}_{anatomy}"
        
        if model_key in self.models:
            self.current_model = self.models[model_key]
            return
        
        if modality == "X-RAY":
            if anatomy == "CHEST":
                model = self._load_chest_xray_model()
            else:
                model = self._load_general_xray_model()
        elif modality == "CT":
            if anatomy == "CHEST":
                model = self._load_chest_ct_model()
            elif anatomy == "HEAD":
                model = self._load_head_ct_model()
            else:
                model = self._load_general_ct_model()
        elif modality == "MRI":
            if anatomy == "BRAIN":
                model = self._load_brain_mri_model()
            else:
                model = self._load_general_mri_model()
        else:
            model = self._load_general_medical_model()
        
        model.to(self.device)
        model.eval()
        
        self.models[model_key] = model
        self.current_model = model
        self.current_modality = modality
    
    def _load_chest_xray_model(self) -> nn.Module:
        model = timm.create_model('convnext_base', pretrained=True, num_classes=15)
        return model
    
    def _load_general_xray_model(self) -> nn.Module:
        model = timm.create_model('resnet50', pretrained=True, num_classes=10)
        return model
    
    def _load_chest_ct_model(self) -> nn.Module:
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=8)
        return model
    
    def _load_head_ct_model(self) -> nn.Module:
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
        return model
    
    def _load_brain_mri_model(self) -> nn.Module:
        model = monai.networks.nets.HighResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4
        )
        return model
    
    def _load_general_medical_model(self) -> nn.Module:
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=20)
        return model
    
    def analyze_image(self, image: torch.Tensor, modality: str, anatomy: str, 
                     clinical_context: Dict[str, Any] = None, confidence_threshold: float = 0.75,
                     enable_segmentation: bool = True) -> Dict[str, Any]:
        self.load_model(modality, anatomy)
        
        with torch.no_grad():
            if len(image.shape) == 4 and image.shape[1] == 1:
                image = image.squeeze(1)
            
            if len(image.shape) == 3:
                image = image.unsqueeze(1)
            
            image = image.to(self.device)
            
            if modality in ["CT", "MRI"] and len(image.shape) == 5:
                classification_output = self._process_volume(image)
            else:
                classification_output = self.current_model(image)
            
            findings = self._interpret_output(classification_output, modality, anatomy, confidence_threshold)
            
            if enable_segmentation and modality in ["CT", "MRI"]:
                segmentation_results = self._perform_segmentation(image, modality, anatomy)
                findings.update(segmentation_results)
            
            findings['confidence'] = self._compute_overall_confidence(findings)
            findings['processing_time'] = 0.0
            
            if clinical_context:
                findings['clinical_context'] = clinical_context
            
            return findings
    
    def _process_volume(self, volume: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = volume.shape
        
        slice_predictions = []
        for i in range(depth):
            slice_data = volume[:, :, i, :, :]
            slice_pred = self.current_model(slice_data)
            slice_predictions.append(slice_pred)
        
        stacked_predictions = torch.stack(slice_predictions, dim=1)
        volume_prediction = torch.mean(stacked_predictions, dim=1)
        
        return volume_prediction
    
    def _interpret_output(self, output: torch.Tensor, modality: str, anatomy: str, 
                         confidence_threshold: float) -> Dict[str, Any]:
        probabilities = torch.softmax(output, dim=1)
        max_probs, predicted_classes = torch.max(probabilities, dim=1)
        
        findings = {
            'primary_findings': [],
            'detailed_findings': {},
            'quantitative_metrics': {}
        }
        
        if modality == "X-RAY" and anatomy == "CHEST":
            pathology_classes = [
                "Normal", "Pneumonia", "Pneumothorax", "Edema", "Cardiomegaly",
                "Effusion", "Nodule", "Fibrosis", "Consolidation", "Atelectasis"
            ]
            
            for i, (prob, class_idx) in enumerate(zip(max_probs[0], predicted_classes[0])):
                if prob > confidence_threshold and class_idx < len(pathology_classes):
                    finding = {
                        'finding': pathology_classes[class_idx],
                        'confidence': float(prob),
                        'location': 'Lungs',
                        'severity': self._assess_severity(prob)
                    }
                    findings['primary_findings'].append(finding)
        
        elif modality == "CT" and anatomy == "CHEST":
            nodule_prob = probabilities[0, 1]
            if nodule_prob > confidence_threshold:
                findings['primary_findings'].append({
                    'finding': 'Pulmonary Nodule',
                    'confidence': float(nodule_prob),
                    'location': 'Lung parenchyma',
                    'severity': self._assess_severity(nodule_prob)
                })
        
        elif modality == "MRI" and anatomy == "BRAIN":
            tumor_prob = probabilities[0, 2]
            if tumor_prob > confidence_threshold:
                findings['primary_findings'].append({
                    'finding': 'Brain Tumor',
                    'confidence': float(tumor_prob),
                    'location': 'Intracranial',
                    'severity': self._assess_severity(tumor_prob)
                })
        
        return findings
    
    def _assess_severity(self, confidence: float) -> str:
        if confidence > 0.85:
            return "SEVERE"
        elif confidence > 0.7:
            return "MODERATE"
        else:
            return "MILD"
    
    def _perform_segmentation(self, image: torch.Tensor, modality: str, anatomy: str) -> Dict[str, Any]:
        segmentation_results = {}
        
        if modality == "CT" and anatomy == "CHEST":
            lung_mask = self._segment_lungs(image)
            segmentation_results['lung_volume'] = float(torch.sum(lung_mask))
            segmentation_results['segmentation_masks'] = {'lungs': lung_mask.cpu().numpy()}
        
        elif modality == "MRI" and anatomy == "BRAIN":
            brain_mask = self._segment_brain(image)
            segmentation_results['brain_volume'] = float(torch.sum(brain_mask))
            segmentation_results['segmentation_masks'] = {'brain': brain_mask.cpu().numpy()}
        
        segmentation_results['segmentation_quality'] = 0.85
        
        return segmentation_results
    
    def _segment_lungs(self, image: torch.Tensor) -> torch.Tensor:
        threshold = -400
        lung_mask = (image > threshold).float()
        
        from scipy import ndimage
        lung_mask_np = lung_mask.cpu().numpy()
        
        for i in range(lung_mask_np.shape[0]):
            labeled_mask, num_features = ndimage.label(lung_mask_np[i, 0])
            if num_features > 0:
                sizes = ndimage.sum(lung_mask_np[i, 0], labeled_mask, range(1, num_features + 1))
                max_label = np.argmax(sizes) + 1
                lung_mask_np[i, 0] = (labeled_mask == max_label).astype(np.float32)
        
        return torch.from_numpy(lung_mask_np).to(self.device)
    
    def _segment_brain(self, image: torch.Tensor) -> torch.Tensor:
        brain_mask = (image > image.mean()).float()
        return brain_mask
    
    def _compute_overall_confidence(self, findings: Dict[str, Any]) -> float:
        if not findings['primary_findings']:
            return 0.95
        
        confidences = [finding['confidence'] for finding in findings['primary_findings']]
        return float(np.mean(confidences))

class AdvancedMedicalPredictor(MedicalPredictor):
    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.uncertainty_estimator = UncertaintyEstimator()
    
    def analyze_with_uncertainty(self, image: torch.Tensor, modality: str, anatomy: str,
                               num_samples: int = 10) -> Dict[str, Any]:
        self.load_model(modality, anatomy)
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.current_model(image)
                predictions.append(pred)
        
        stacked_preds = torch.stack(predictions)
        mean_prediction = torch.mean(stacked_preds, dim=0)
        uncertainty = torch.std(stacked_preds, dim=0)
        
        findings = self._interpret_output(mean_prediction, modality, anatomy, 0.5)
        findings['uncertainty'] = float(torch.mean(uncertainty))
        findings['confidence_intervals'] = self._compute_confidence_intervals(stacked_preds)
        
        return findings
    
    def _compute_confidence_intervals(self, predictions: torch.Tensor) -> Dict[str, Any]:
        lower_quantile = torch.quantile(predictions, 0.025, dim=0)
        upper_quantile = torch.quantile(predictions, 0.975, dim=0)
        
        return {
            'lower_bound': lower_quantile.cpu().numpy(),
            'upper_bound': upper_quantile.cpu().numpy()
        }

class UncertaintyEstimator:
    def __init__(self):
        pass
    
    def estimate_aleatoric(self, predictions: torch.Tensor) -> torch.Tensor:
        return torch.var(predictions, dim=0)
    
    def estimate_epistemic(self, model: nn.Module, image: torch.Tensor, 
                          num_samples: int = 10) -> torch.Tensor:
        uncertainties = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                pred = model(image)
                uncertainties.append(pred)
        
        stacked_uncertainties = torch.stack(uncertainties)
        return torch.std(stacked_uncertainties, dim=0)