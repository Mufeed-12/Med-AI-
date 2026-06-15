import numpy as np
from typing import Dict, Any, List, Tuple
import torch

class MedicalMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_dice_score(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        intersection = np.sum(prediction * ground_truth)
        union = np.sum(prediction) + np.sum(ground_truth)
        
        if union == 0:
            return 1.0
        
        return 2.0 * intersection / union
    
    def calculate_hausdorff_distance(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        from scipy.spatial.distance import directed_hausdorff
        
        pred_points = np.argwhere(prediction > 0)
        gt_points = np.argwhere(ground_truth > 0)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return float('inf')
        
        hausdorff_1 = directed_hausdorff(pred_points, gt_points)[0]
        hausdorff_2 = directed_hausdorff(gt_points, pred_points)[0]
        
        return max(hausdorff_1, hausdorff_2)
    
    def calculate_roc_metrics(self, predictions: List[float], ground_truth: List[int]) -> Dict[str, float]:
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        roc_auc = auc(fpr, tpr)
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    def calculate_clinical_metrics(self, predictions: Dict[str, Any], 
                                 ground_truth: Dict[str, Any]) -> Dict[str, float]:
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        pred_findings = set([f['finding'] for f in predictions.get('primary_findings', [])])
        gt_findings = set([f['finding'] for f in ground_truth.get('primary_findings', [])])
        
        for finding in pred_findings:
            if finding in gt_findings:
                true_positives += 1
            else:
                false_positives += 1
        
        for finding in gt_findings:
            if finding not in pred_findings:
                false_negatives += 1
        
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        quality_metrics = {}
        
        quality_metrics['signal_to_noise'] = self._calculate_snr(image)
        quality_metrics['contrast_to_noise'] = self._calculate_cnr(image)
        quality_metrics['uniformity'] = self._calculate_uniformity(image)
        quality_metrics['sharpness'] = self._calculate_sharpness(image)
        
        overall_quality = np.mean(list(quality_metrics.values()))
        quality_metrics['overall_quality_score'] = overall_quality
        
        return quality_metrics
    
    def _calculate_snr(self, image: np.ndarray) -> float:
        signal_mean = np.mean(image)
        noise_std = np.std(image)
        
        if noise_std == 0:
            return float('inf')
        
        return signal_mean / noise_std
    
    def _calculate_cnr(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            image = image[image.shape[0] // 2]
        
        quarter_h = image.shape[0] // 4
        quarter_w = image.shape[1] // 4
        
        center_roi = image[quarter_h:3*quarter_h, quarter_w:3*quarter_w]
        corner_roi = image[:quarter_h, :quarter_w]
        
        contrast = np.mean(center_roi) - np.mean(corner_roi)
        noise = np.std(image)
        
        if noise == 0:
            return 0
        
        return abs(contrast) / noise
    
    def _calculate_uniformity(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            image = image[image.shape[0] // 2]
        
        center_region = image[image.shape[0]//4:3*image.shape[0]//4,
                            image.shape[1]//4:3*image.shape[1]//4]
        
        return 1.0 - (np.std(center_region) / (np.mean(center_region) + 1e-8))
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        from scipy import ndimage
        
        if len(image.shape) == 3:
            image = image[image.shape[0] // 2]
        
        sobel_x = ndimage.sobel(image, axis=0)
        sobel_y = ndimage.sobel(image, axis=1)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return np.mean(gradient_magnitude)

class AdvancedMedicalMetrics(MedicalMetrics):
    def __init__(self):
        super().__init__()
    
    def calculate_uncertainty_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        confidences = [pred.get('confidence', 0) for pred in predictions]
        
        if not confidences:
            return {'uncertainty': 0, 'confidence_interval': 0}
        
        uncertainty = np.std(confidences)
        mean_confidence = np.mean(confidences)
        
        confidence_interval = 1.96 * uncertainty / np.sqrt(len(confidences))
        
        return {
            'uncertainty': uncertainty,
            'mean_confidence': mean_confidence,
            'confidence_interval': confidence_interval,
            'confidence_range': [mean_confidence - confidence_interval, mean_confidence + confidence_interval]
        }
    
    def assess_clinical_significance(self, findings: Dict[str, Any], 
                                   clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        
        significance_metrics = {}
        
        primary_findings = findings.get('primary_findings', [])
        
        critical_count = len([f for f in primary_findings if f.get('severity') in ['SEVERE', 'MODERATE']])
        significance_metrics['critical_finding_ratio'] = critical_count / len(primary_findings) if primary_findings else 0
        
        avg_confidence = np.mean([f['confidence'] for f in primary_findings]) if primary_findings else 0
        significance_metrics['average_confidence'] = avg_confidence
        
        age = clinical_context.get('age', 45)
        age_risk_factor = min(age / 80, 1.0)
        significance_metrics['age_adjusted_risk'] = age_risk_factor
        
        overall_significance = (significance_metrics['critical_finding_ratio'] * 0.4 +
                              significance_metrics['average_confidence'] * 0.3 +
                              significance_metrics['age_adjusted_risk'] * 0.3)
        
        significance_metrics['overall_clinical_significance'] = overall_significance
        
        return significance_metrics