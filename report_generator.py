import datetime
from typing import Dict, Any, List
from fpdf import FPDF
import json

class ReportGenerator:
    def __init__(self):
        self.report_templates = {
            'radiology_standard': self._radiology_standard_template,
            'clinical_summary': self._clinical_summary_template,
            'research_protocol': self._research_protocol_template
        }
    
    def generate_report(self, findings: Dict[str, Any], correlated_findings: Dict[str, Any] = None,
                       patient_data: Dict[str, Any] = None, study_info: Dict[str, Any] = None) -> Dict[str, Any]:
        
        template = self.report_templates['radiology_standard']
        report_text = template(findings, correlated_findings, patient_data, study_info)
        
        recommendations = self._generate_recommendations(findings, patient_data)
        
        report_data = {
            'full_report': report_text,
            'summary': self._generate_summary(findings),
            'recommendations': recommendations,
            'findings_breakdown': self._categorize_findings(findings),
            'timestamp': datetime.datetime.now().isoformat(),
            'report_id': self._generate_report_id()
        }
        
        return report_data
    
    def _radiology_standard_template(self, findings: Dict[str, Any], correlated_findings: Dict[str, Any],
                                   patient_data: Dict[str, Any], study_info: Dict[str, Any]) -> str:
        
        report_parts = []
        
        report_parts.append("RADIOLOGY REPORT")
        report_parts.append("=" * 50)
        
        if patient_data:
            report_parts.append(f"Patient: {patient_data.get('id', 'N/A')}")
            report_parts.append(f"Age: {patient_data.get('age', 'N/A')}")
            report_parts.append(f"Gender: {patient_data.get('gender', 'N/A')}")
        
        if study_info:
            report_parts.append(f"Modality: {study_info.get('modality', 'N/A')}")
            report_parts.append(f"Body Part: {study_info.get('body_part', 'N/A')}")
            report_parts.append(f"Study Date: {study_info.get('date', 'N/A')}")
        
        report_parts.append("\nCLINICAL HISTORY:")
        if patient_data and 'clinical_notes' in patient_data:
            report_parts.append(patient_data['clinical_notes'])
        else:
            report_parts.append("Not provided")
        
        report_parts.append("\nFINDINGS:")
        if findings.get('primary_findings'):
            for finding in findings['primary_findings']:
                report_parts.append(f"- {finding['finding']} (Confidence: {finding['confidence']:.2f})")
                report_parts.append(f"  Location: {finding.get('location', 'N/A')}")
                report_parts.append(f"  Severity: {finding.get('severity', 'N/A')}")
        else:
            report_parts.append("No significant abnormalities detected.")
        
        if findings.get('quantitative_metrics'):
            report_parts.append("\nQUANTITATIVE ANALYSIS:")
            for metric, value in findings['quantitative_metrics'].items():
                report_parts.append(f"- {metric}: {value}")
        
        report_parts.append(f"\nOVERALL CONFIDENCE: {findings.get('confidence', 0):.2f}")
        
        if correlated_findings:
            report_parts.append("\nCLINICAL CORRELATION:")
            for correlation in correlated_findings.get('clinical_correlations', []):
                report_parts.append(f"- {correlation}")
        
        report_parts.append("\nIMPRESSION:")
        impression = self._generate_impression(findings, correlated_findings)
        report_parts.append(impression)
        
        return "\n".join(report_parts)
    
    def _clinical_summary_template(self, findings: Dict[str, Any], correlated_findings: Dict[str, Any],
                                 patient_data: Dict[str, Any], study_info: Dict[str, Any]) -> str:
        
        summary_parts = []
        
        summary_parts.append("CLINICAL SUMMARY REPORT")
        summary_parts.append("=" * 40)
        
        critical_findings = [f for f in findings.get('primary_findings', []) 
                           if f.get('severity') in ['SEVERE', 'MODERATE']]
        
        if critical_findings:
            summary_parts.append("\nCRITICAL FINDINGS:")
            for finding in critical_findings:
                summary_parts.append(f"- {finding['finding']} ({finding['severity']})")
        else:
            summary_parts.append("\nNo critical findings detected.")
        
        summary_parts.append(f"\nDiagnostic Confidence: {findings.get('confidence', 0):.2f}")
        
        return "\n".join(summary_parts)
    
    def _research_protocol_template(self, findings: Dict[str, Any], correlated_findings: Dict[str, Any],
                                  patient_data: Dict[str, Any], study_info: Dict[str, Any]) -> str:
        
        research_parts = []
        
        research_parts.append("RESEARCH PROTOCOL REPORT")
        research_parts.append("=" * 40)
        
        research_parts.append(f"Total Findings: {len(findings.get('primary_findings', []))}")
        research_parts.append(f"Average Confidence: {findings.get('confidence', 0):.2f}")
        
        if findings.get('quantitative_metrics'):
            research_parts.append("\nQUANTITATIVE METRICS:")
            for metric, value in findings['quantitative_metrics'].items():
                research_parts.append(f"{metric}: {value}")
        
        return "\n".join(research_parts)
    
    def _generate_recommendations(self, findings: Dict[str, Any], patient_data: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        primary_findings = findings.get('primary_findings', [])
        
        for finding in primary_findings:
            finding_name = finding['finding']
            severity = finding.get('severity', 'MILD')
            
            if finding_name == "Pulmonary Nodule":
                if severity == "SEVERE":
                    recommendations.append("Recommend CT follow-up in 3-6 months and consider PET-CT for further characterization.")
                else:
                    recommendations.append("Recommend routine follow-up based on Fleischner Society guidelines.")
            
            elif finding_name == "Pneumonia":
                recommendations.append("Consider antibiotic therapy and follow-up chest imaging in 4-6 weeks.")
            
            elif finding_name == "Brain Tumor":
                recommendations.append("Urgent neurosurgery consultation recommended. Consider contrast-enhanced MRI.")
            
            elif finding_name == "Fracture":
                recommendations.append("Orthopedic consultation recommended for fracture management.")
        
        if not primary_findings:
            recommendations.append("No additional imaging recommended at this time.")
        
        patient_age = patient_data.get('age', 0) if patient_data else 0
        if patient_age > 50 and not primary_findings:
            recommendations.append("Consider age-appropriate screening based on clinical guidelines.")
        
        return recommendations
    
    def _generate_summary(self, findings: Dict[str, Any]) -> str:
        primary_findings = findings.get('primary_findings', [])
        
        if not primary_findings:
            return "Normal study. No significant abnormalities detected."
        
        critical_count = len([f for f in primary_findings if f.get('severity') in ['SEVERE', 'MODERATE']])
        
        if critical_count > 0:
            return f"Abnormal study with {critical_count} significant finding(s) requiring clinical attention."
        else:
            return f"Study with {len(primary_findings)} minor finding(s). Clinical correlation recommended."
    
    def _categorize_findings(self, findings: Dict[str, Any]) -> Dict[str, List]:
        categorized = {
            'critical': [],
            'moderate': [],
            'minor': []
        }
        
        for finding in findings.get('primary_findings', []):
            severity = finding.get('severity', 'MILD')
            
            if severity == 'SEVERE':
                categorized['critical'].append(finding)
            elif severity == 'MODERATE':
                categorized['moderate'].append(finding)
            else:
                categorized['minor'].append(finding)
        
        return categorized
    
    def _generate_impression(self, findings: Dict[str, Any], correlated_findings: Dict[str, Any]) -> str:
        primary_findings = findings.get('primary_findings', [])
        
        if not primary_findings:
            return "Within normal limits for age and clinical indication."
        
        critical_findings = [f for f in primary_findings if f.get('severity') in ['SEVERE', 'MODERATE']]
        
        if critical_findings:
            finding_names = [f['finding'] for f in critical_findings]
            return f"Findings consistent with {', '.join(finding_names)}. Clinical correlation and appropriate follow-up recommended."
        else:
            return "Minor findings noted as above. No acute abnormality identified."
    
    def _generate_report_id(self) -> str:
        import uuid
        return f"RPT_{uuid.uuid4().hex[:8].upper()}"

class PDFReportGenerator(ReportGenerator):
    def __init__(self):
        super().__init__()
    
    def generate_pdf_report(self, findings: Dict[str, Any], patient_data: Dict[str, Any],
                          study_info: Dict[str, Any], output_path: str):
        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "MEDVISION AI - RADIOLOGY REPORT", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("Arial", '', 12)
        
        if patient_data:
            pdf.cell(0, 10, f"Patient ID: {patient_data.get('id', 'N/A')}", 0, 1)
            pdf.cell(0, 10, f"Age: {patient_data.get('age', 'N/A')} | Gender: {patient_data.get('gender', 'N/A')}", 0, 1)
        
        if study_info:
            pdf.cell(0, 10, f"Modality: {study_info.get('modality', 'N/A')} | Body Part: {study_info.get('body_part', 'N/A')}", 0, 1)
        
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "FINDINGS:", 0, 1)
        pdf.set_font("Arial", '', 12)
        
        if findings.get('primary_findings'):
            for finding in findings['primary_findings']:
                pdf.multi_cell(0, 10, f"- {finding['finding']} (Confidence: {finding['confidence']:.2f}, Severity: {finding.get('severity', 'N/A')})")
        else:
            pdf.cell(0, 10, "No significant abnormalities detected.", 0, 1)
        
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "IMPRESSION:", 0, 1)
        pdf.set_font("Arial", '', 12)
        
        impression = self._generate_impression(findings, None)
        pdf.multi_cell(0, 10, impression)
        
        pdf.output(output_path)