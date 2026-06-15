import pydicom
import SimpleITK as sitk
import numpy as np
import torch
from typing import Dict, Any, Optional
import tempfile
import os

class DICOMProcessor:
    def __init__(self):
        self.supported_modalities = ['CT', 'MR', 'XR', 'PT', 'US']
    
    def load_dicom(self, file_path: str) -> Dict[str, Any]:
        try:
            if file_path.lower().endswith(('.dcm', '.dicom')):
                return self._load_single_dicom(file_path)
            elif file_path.lower().endswith(('.nii', '.nii.gz')):
                return self._load_nifti(file_path)
            else:
                return self._load_standard_image(file_path)
        except Exception as e:
            raise Exception(f"Failed to load medical image: {str(e)}")
    
    def _load_single_dicom(self, file_path: str) -> Dict[str, Any]:
        dataset = pydicom.dcmread(file_path)
        
        image_data = self._extract_pixel_data(dataset)
        
        metadata = {
            'modality': getattr(dataset, 'Modality', 'UNKNOWN'),
            'body_part': getattr(dataset, 'BodyPartExamined', 'UNKNOWN'),
            'study_date': getattr(dataset, 'StudyDate', ''),
            'series_description': getattr(dataset, 'SeriesDescription', ''),
            'patient_position': getattr(dataset, 'PatientPosition', ''),
            'pixel_spacing': getattr(dataset, 'PixelSpacing', [1.0, 1.0]),
            'slice_thickness': getattr(dataset, 'SliceThickness', 1.0)
        }
        
        return {
            'image_data': image_data,
            'metadata': metadata,
            'original_dataset': dataset
        }
    
    def _load_nifti(self, file_path: str) -> Dict[str, Any]:
        image = sitk.ReadImage(file_path)
        image_data = sitk.GetArrayFromImage(image)
        
        metadata = {
            'modality': 'UNKNOWN',
            'body_part': 'UNKNOWN',
            'pixel_spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection()
        }
        
        return {
            'image_data': image_data,
            'metadata': metadata,
            'original_image': image
        }
    
    def _load_standard_image(self, file_path: str) -> Dict[str, Any]:
        import cv2
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not load standard image file")
        
        metadata = {
            'modality': 'XR',
            'body_part': 'UNKNOWN',
            'pixel_spacing': [1.0, 1.0]
        }
        
        return {
            'image_data': image,
            'metadata': metadata
        }
    
    def _extract_pixel_data(self, dataset) -> np.ndarray:
        pixel_data = dataset.pixel_array
        
        if 'WindowWidth' in dataset and 'WindowCenter' in dataset:
            window_center = dataset.WindowCenter
            window_width = dataset.WindowWidth
            
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]
            
            pixel_data = self._apply_window_level(pixel_data, window_center, window_width)
        
        return pixel_data.astype(np.float32)
    
    def _apply_window_level(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        min_val = center - width / 2
        max_val = center + width / 2
        
        windowed = np.clip(image, min_val, max_val)
        normalized = (windowed - min_val) / (max_val - min_val)
        
        return normalized
    
    def preprocess(self, dicom_data: Dict[str, Any]) -> torch.Tensor:
        image_data = dicom_data['image_data']
        metadata = dicom_data['metadata']
        
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=0)
        
        normalized = self._normalize_intensity(image_data)
        
        if metadata['modality'] in ['CT', 'MR'] and len(normalized.shape) == 3:
            resampled = self._resample_volume(normalized, metadata.get('pixel_spacing', [1.0, 1.0, 1.0]))
        else:
            resampled = normalized
        
        tensor_data = torch.from_numpy(resampled).float()
        
        if len(tensor_data.shape) == 3:
            tensor_data = tensor_data.unsqueeze(0)
        
        return tensor_data
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        non_zero_mask = image > 0
        if np.any(non_zero_mask):
            mean_val = np.mean(image[non_zero_mask])
            std_val = np.std(image[non_zero_mask])
            if std_val > 0:
                normalized = (image - mean_val) / std_val
            else:
                normalized = image - mean_val
        else:
            normalized = image
        
        return normalized
    
    def _resample_volume(self, volume: np.ndarray, spacing: list) -> np.ndarray:
        target_spacing = [1.0, 1.0, 1.0]
        
        if len(spacing) == 2:
            spacing = [spacing[0], spacing[1], 1.0]
        
        scale_factors = [spacing[i] / target_spacing[i] for i in range(3)]
        
        new_shape = [int(volume.shape[i] * scale_factors[i]) for i in range(3)]
        
        import scipy.ndimage
        resampled = scipy.ndimage.zoom(volume, scale_factors, order=1)
        
        return resampled

class AdvancedDICOMProcessor(DICOMProcessor):
    def __init__(self):
        super().__init__()
    
    def detect_artifacts(self, image_data: np.ndarray) -> Dict[str, Any]:
        artifacts = {}
        
        if len(image_data.shape) == 3:
            slice_artifacts = self._analyze_volume_artifacts(image_data)
            artifacts.update(slice_artifacts)
        
        noise_level = self._estimate_noise_level(image_data)
        artifacts['noise_level'] = noise_level
        
        uniformity = self._assess_uniformity(image_data)
        artifacts['uniformity'] = uniformity
        
        return artifacts
    
    def _analyze_volume_artifacts(self, volume: np.ndarray) -> Dict[str, Any]:
        artifacts = {}
        
        slice_correlations = []
        for i in range(1, volume.shape[0]):
            corr = np.corrcoef(volume[i-1].flatten(), volume[i].flatten())[0,1]
            slice_correlations.append(corr)
        
        mean_correlation = np.mean(slice_correlations)
        artifacts['slice_consistency'] = mean_correlation
        
        return artifacts
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        from scipy import ndimage
        
        if len(image.shape) == 3:
            image = image[image.shape[0] // 2]
        
        smooth = ndimage.gaussian_filter(image, sigma=1)
        noise = image - smooth
        
        return np.std(noise)
    
    def _assess_uniformity(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            middle_slice = image[image.shape[0] // 2]
        else:
            middle_slice = image
        
        center_region = middle_slice[middle_slice.shape[0]//4:3*middle_slice.shape[0]//4,
                                   middle_slice.shape[1]//4:3*middle_slice.shape[1]//4]
        
        return np.std(center_region) / (np.mean(center_region) + 1e-8)