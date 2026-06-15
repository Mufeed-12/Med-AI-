from typing import Dict, Any, List
import numpy as np

class ClinicalCorrelator:
    def __init__(self):
        self.clinical_knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        return {
            'age_related_findings': {
                'pediatric': ['Congenital anomalies', 'Growth-related changes'],
                'adult': ['Degenerative changes', 'Early onset diseases'],
                'geriatric': ['Advanced degeneration', 'Age-related pathologies']
            },
            'gender_specific': {
                'Male': ['Prostate conditions', 'Testicular pathologies'],
                'Female': ['Ovarian conditions', 'Breast pathologies', 'Uterine findings']
            },
            'risk_factors': {
                'smoking': ['Lung cancer', 'COPD', 'Emphysema'],
                'hypertension': ['Cardiomegaly', 'Vascular changes'],
                'diabetes': ['Vascular calcifications', 'Infections']
            }
        }
    
    def correlate_findings(self, findings: Dict[str, Any], clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        correlated_results = {
            'clinical_correlations': [],
            'risk_assessment': {},
            'differential_diagnosis': [],
            'contextual_interpretation': {}
        }
        
        age = clinical_context.get('age', 0)
        gender = clinical_context.get('gender', '')
        clinical_notes = clinical_context.get('notes', '')
        
        correlated_results['clinical_correlations'].extend(
            self._correlate_with_age(findings, age)
        )
        
        correlated_results['clinical_correlations'].extend(
            self._correlate_with_gender(findings, gender)
        )
        
        correlated_results['risk_assessment'] = self._assess_risk_level(findings, clinical_context)
        
        correlated_results['differential_diagnosis'] = self._generate_differential_diagnosis(
            findings, clinical_context
        )
        
        correlated_results['contextual_interpretation'] = self._interpret_in_context(
            findings, clinical_context
        )
        
        return correlated_results
    
    def _correlate_with_age(self, findings: Dict[str, Any], age: int) -> List[str]:
        correlations = []
        
        age_group = self._get_age_group(age)
        age_related = self.clinical_knowledge_base['age_related_findings'].get(age_group, [])
        
        primary_findings = [f['finding'] for f in findings.get('primary_findings', [])]
        
        for finding in primary_findings:
            if any(age_pathology in finding for age_pathology in age_related):
                correlations.append(f"Finding '{finding}' is commonly seen in {age_group} population")
        
        return correlations
    
    def _correlate_with_gender(self, findings: Dict[str, Any], gender: str) -> List[str]:
        correlations = []
        
        if not gender:
            return correlations
        
        gender_specific = self.clinical_knowledge_base['gender_specific'].get(gender, [])
        primary_findings = [f['finding'] for f in findings.get('primary_findings', [])]
        
        for finding in primary_findings:
            if any(gender_pathology in finding for gender_pathology in gender_specific):
                correlations.append(f"Finding '{finding}' has gender-specific considerations for {gender}")
        
        return correlations
    
    def _assess_risk_level(self, findings: Dict[str, Any], clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        risk_factors = self._extract_risk_factors(clinical_context.get('notes', ''))
        
        severity_scores = {
            'SEVERE': 3,
            'MODERATE': 2, 
            'MILD': 1
        }
        
        total_risk_score = 0
        critical_findings = 0
        
        for finding in findings.get('primary_findings', []):
            severity = finding.get('severity', 'MILD')
            total_risk_score += severity_scores.get(severity, 1)
            if severity in ['SEVERE', 'MODERATE']:
                critical_findings += 1
        
        risk_level = "LOW"
        if total_risk_score >= 6:
            risk_level = "HIGH"
        elif total_risk_score >= 3:
            risk_level = "MODERATE"
        
        return {
            'risk_level': risk_level,
            'risk_score': total_risk_score,
            'critical_findings_count': critical_findings,
            'identified_risk_factors': risk_factors
        }
    
    def _generate_differential_diagnosis(self, findings: Dict[str, Any], 
                                       clinical_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        differentials = []
        primary_findings = findings.get('primary_findings', [])
        
        if not primary_findings:
            return differentials
        
        finding_names = [f['finding'] for f in primary_findings]
        
        differential_knowledge = {
            'Pulmonary Nodule': [
                {'diagnosis': 'Primary Lung Cancer', 'probability': 0.3},
                {'diagnosis': 'Metastasis', 'probability': 0.25},
                {'diagnosis': 'Benign Granuloma', 'probability': 0.2},
                {'diagnosis': 'Infection', 'probability': 0.15},
                {'diagnosis': 'Hamartoma', 'probability': 0.1}
            ],
            'Pneumonia': [
                {'diagnosis': 'Bacterial Pneumonia', 'probability': 0.4},
                {'diagnosis': 'Viral Pneumonia', 'probability': 0.3},
                {'diagnosis': 'Atypical Pneumonia', 'probability': 0.2},
                {'diagnosis': 'Fungal Infection', 'probability': 0.1}
            ],
            'Brain Tumor': [
                {'diagnosis': 'Glioma', 'probability': 0.35},
                {'diagnosis': 'Meningioma', 'probability': 0.25},
                {'diagnosis': 'Metastasis', 'probability': 0.2},
                {'diagnosis': 'Lymphoma', 'probability': 0.1},
                {'diagnosis': 'Other', 'probability': 0.1}
            ]
        }
        
        for finding_name in finding_names:
            if finding_name in differential_knowledge:
                differentials.extend(differential_knowledge[finding_name])
        
        return sorted(differentials, key=lambda x: x['probability'], reverse=True)[:5]
    
    def _interpret_in_context(self, findings: Dict[str, Any], clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        interpretation = {
            'clinical_significance': 'UNCERTAIN',
            'urgency_level': 'ROUTINE',
            'followup_timeline': 'STANDARD'
        }
        
        risk_assessment = self._assess_risk_level(findings, clinical_context)
        
        if risk_assessment['risk_level'] == 'HIGH':
            interpretation['clinical_significance'] = 'HIGH'
            interpretation['urgency_level'] = 'URGENT'
            interpretation['followup_timeline'] = 'IMMEDIATE'
        elif risk_assessment['risk_level'] == 'MODERATE':
            interpretation['clinical_significance'] = 'MODERATE'
            interpretation['urgency_level'] = 'PRIORITY'
            interpretation['followup_timeline'] = 'SHORT_TERM'
        else:
            interpretation['clinical_significance'] = 'LOW'
            interpretation['urgency_level'] = 'ROUTINE'
            interpretation['followup_timeline'] = 'ROUTINE'
        
        return interpretation
    
    def _get_age_group(self, age: int) -> str:
        if age < 18:
            return 'pediatric'
        elif age < 65:
            return 'adult'
        else:
            return 'geriatric'
    
    def _extract_risk_factors(self, clinical_notes: str) -> List[str]:
        risk_keywords = {
            'smoking': ['smoking', 'smoker', 'tobacco'],
            'hypertension': ['hypertension', 'high blood pressure'],
            'diabetes': ['diabetes', 'diabetic'],
            'obesity': ['obesity', 'obese', 'overweight'],
            'family_history': ['family history', 'familial']
        }
        
        identified_factors = []
        notes_lower = clinical_notes.lower()
        
        for factor, keywords in risk_keywords.items():
            if any(keyword in notes_lower for keyword in keywords):
                identified_factors.append(factor)
        
        return identified_factors

class AdvancedClinicalCorrelator(ClinicalCorrelator):
    def __init__(self):
        super().__init__()
        self.evidence_levels = self._initialize_evidence_levels()
    
    def _initialize_evidence_levels(self) -> Dict[str, int]:
        return {
            'randomized_trial': 5,
            'cohort_study': 4,
            'case_control': 3,
            'case_series': 2,
            'expert_opinion': 1
        }
    
    def correlate_with_evidence(self, findings: Dict[str, Any], clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        basic_correlation = self.correlate_findings(findings, clinical_context)
        
        evidence_based_interpretation = self._apply_evidence_levels(
            basic_correlation, findings, clinical_context
        )
        
        basic_correlation['evidence_based_interpretation'] = evidence_based_interpretation
        
        return basic_correlation
    
    def _apply_evidence_levels(self, correlation: Dict[str, Any], findings: Dict[str, Any],
                             clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        
        evidence_interpretation = {
            'evidence_level': 'EXPERT_OPINION',
            'supporting_studies': [],
            'confidence_in_evidence': 0.0
        }
        
        primary_findings = findings.get('primary_findings', [])
        
        if not primary_findings:
            evidence_interpretation['evidence_level'] = 'NO_EVIDENCE_NEEDED'
            evidence_interpretation['confidence_in_evidence'] = 0.95
            return evidence_interpretation
        
        finding_evidence_levels = []
        
        for finding in primary_findings:
            finding_name = finding['finding']
            evidence_level = self._get_evidence_level_for_finding(finding_name)
            finding_evidence_levels.append(evidence_level)
        
        if finding_evidence_levels:
            avg_evidence_level = sum(finding_evidence_levels) / len(finding_evidence_levels)
            evidence_interpretation['confidence_in_evidence'] = avg_evidence_level / 5.0
            
            if avg_evidence_level >= 4:
                evidence_interpretation['evidence_level'] = 'HIGH_QUALITY_EVIDENCE'
            elif avg_evidence_level >= 3:
                evidence_interpretation['evidence_level'] = 'MODERATE_EVIDENCE'
            else:
                evidence_interpretation['evidence_level'] = 'LOW_EVIDENCE'
        
        return evidence_interpretation
    
    def _get_evidence_level_for_finding(self, finding: str) -> int:
        evidence_mapping = {
            'Pulmonary Nodule': 4,
            'Pneumonia': 5,
            'Brain Tumor': 4,
            'Fracture': 5,
            'Cardiomegaly': 4,
            'Pneumothorax': 5
        }
        
        return evidence_mapping.get(finding, 2)