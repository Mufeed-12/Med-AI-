import streamlit as st
import torch
import os
from pathlib import Path
import tempfile
from core.dicom_processor import DICOMProcessor
from core.predictor import MedicalPredictor
from core.model_manager import ModelManager
from core.report_generator import ReportGenerator
from core.clinical_correlator import ClinicalCorrelator
from utils.visualization import MedicalVisualizer
from utils.config import load_config

st.set_page_config(
    page_title="MedVision AI - Medical Imaging Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'dicom_processor' not in st.session_state:
        st.session_state.dicom_processor = None
    if 'medical_predictor' not in st.session_state:
        st.session_state.medical_predictor = None
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'current_study' not in st.session_state:
        st.session_state.current_study = None

def load_models():
    with st.spinner("🔄 Loading medical AI models..."):
        if st.session_state.dicom_processor is None:
            st.session_state.dicom_processor = DICOMProcessor()
        if st.session_state.medical_predictor is None:
            st.session_state.medical_predictor = MedicalPredictor()
        if st.session_state.model_manager is None:
            st.session_state.model_manager = ModelManager()
        if st.session_state.report_generator is None:
            st.session_state.report_generator = ReportGenerator()

def main():
    st.title("🏥 MedVision AI - Medical Imaging Diagnosis Assistant")
    st.markdown("Expert-level medical image analysis with AI-powered diagnostic support")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Clinical Configuration")
        
        modality = st.selectbox(
            "Imaging Modality",
            ["X-Ray", "CT Scan", "MRI"],
            help="Select the medical imaging modality"
        )
        
        anatomy_options = {
            "X-Ray": ["Chest", "Abdomen", "Extremities", "Spine"],
            "CT Scan": ["Head", "Chest", "Abdomen", "Pelvis", "Extremities"],
            "MRI": ["Brain", "Spine", "Abdomen", "Pelvis", "Extremities"]
        }
        
        selected_anatomy = st.selectbox(
            "Anatomical Region",
            anatomy_options[modality]
        )
        
        st.subheader("Clinical Context")
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        clinical_notes = st.text_area("Clinical History", "Patient presents with...")
        
        st.subheader("Analysis Parameters")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
        enable_segmentation = st.checkbox("Enable Anatomical Segmentation", value=True)
        enable_clinical_correlation = st.checkbox("Clinical Correlation", value=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Image Analysis", "🔍 Findings", "📋 Clinical Report", "📈 Quality Metrics"])
    
    with tab1:
        st.header("Medical Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload Medical Image",
            type=['dcm', 'png', 'jpg', 'jpeg', 'nii', 'nii.gz'],
            help="Upload DICOM, NIfTI, or standard image formats"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🩺 Analyze Image", type="primary"):
                    analyze_medical_image(
                        file_path,
                        modality,
                        selected_anatomy,
                        patient_age,
                        patient_gender,
                        clinical_notes,
                        confidence_threshold,
                        enable_segmentation,
                        enable_clinical_correlation
                    )
            
            with col2:
                if st.button("📊 Generate Full Report"):
                    generate_comprehensive_report(
                        file_path,
                        modality,
                        selected_anatomy,
                        patient_age,
                        patient_gender,
                        clinical_notes
                    )
            
            if st.session_state.analysis_results:
                display_analysis_results()
    
    with tab2:
        st.header("Diagnostic Findings")
        if st.session_state.analysis_results:
            display_findings_details()
        else:
            st.info("No analysis performed yet. Upload and analyze an image first.")
    
    with tab3:
        st.header("Clinical Report")
        if st.session_state.analysis_results:
            display_clinical_report()
        else:
            st.info("Generate analysis to view clinical report")
    
    with tab4:
        st.header("Quality Metrics")
        if st.session_state.analysis_results:
            display_quality_metrics()
        else:
            st.info("Quality metrics will appear after analysis")

def analyze_medical_image(file_path, modality, anatomy, age, gender, clinical_notes, confidence_threshold, enable_segmentation, enable_correlation):
    load_models()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Processing medical image...")
        progress_bar.progress(20)
        
        dicom_data = st.session_state.dicom_processor.load_dicom(file_path)
        processed_image = st.session_state.dicom_processor.preprocess(dicom_data)
        progress_bar.progress(40)
        
        status_text.text("🔍 Running AI analysis...")
        predictions = st.session_state.medical_predictor.analyze_image(
            image=processed_image,
            modality=modality.upper(),
            anatomy=anatomy.upper(),
            clinical_context={
                "age": age,
                "gender": gender,
                "clinical_notes": clinical_notes
            },
            confidence_threshold=confidence_threshold,
            enable_segmentation=enable_segmentation
        )
        progress_bar.progress(80)
        
        if enable_correlation:
            status_text.text("📋 Correlating clinical context...")
            clinical_correlator = ClinicalCorrelator()
            correlated_findings = clinical_correlator.correlate_findings(
                findings=predictions,
                clinical_context={
                    "age": age,
                    "gender": gender,
                    "notes": clinical_notes
                }
            )
            predictions['correlated_findings'] = correlated_findings
        
        st.session_state.analysis_results = predictions
        st.session_state.current_study = {
            'file_path': file_path,
            'modality': modality,
            'anatomy': anatomy,
            'patient_info': {'age': age, 'gender': gender}
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"❌ Medical image analysis failed: {str(e)}")

def generate_comprehensive_report(file_path, modality, anatomy, age, gender, clinical_notes):
    load_models()
    
    with st.spinner("📋 Generating comprehensive clinical report..."):
        try:
            dicom_data = st.session_state.dicom_processor.load_dicom(file_path)
            processed_image = st.session_state.dicom_processor.preprocess(dicom_data)
            
            predictions = st.session_state.medical_predictor.analyze_image(
                image=processed_image,
                modality=modality.upper(),
                anatomy=anatomy.upper(),
                clinical_context={
                    "age": age,
                    "gender": gender,
                    "clinical_notes": clinical_notes
                },
                confidence_threshold=0.7,
                enable_segmentation=True
            )
            
            clinical_correlator = ClinicalCorrelator()
            correlated_findings = clinical_correlator.correlate_findings(
                findings=predictions,
                clinical_context={
                    "age": age,
                    "gender": gender,
                    "notes": clinical_notes
                }
            )
            
            report = st.session_state.report_generator.generate_report(
                findings=predictions,
                correlated_findings=correlated_findings,
                patient_data={
                    "id": "STUDY_001",
                    "age": age,
                    "gender": gender,
                    "clinical_notes": clinical_notes
                },
                study_info={
                    "modality": modality,
                    "body_part": anatomy,
                    "date": "2024-01-01"
                }
            )
            
            st.session_state.analysis_results = predictions
            st.session_state.current_study = {
                'file_path': file_path,
                'modality': modality,
                'anatomy': anatomy,
                'patient_info': {'age': age, 'gender': gender}
            }
            
            st.success("✅ Comprehensive report generated!")
            
        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")

def display_analysis_results():
    results = st.session_state.analysis_results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primary Findings")
        if results.get('primary_findings'):
            for finding in results['primary_findings']:
                confidence_color = "🟢" if finding['confidence'] > 0.8 else "🟡" if finding['confidence'] > 0.6 else "🔴"
                st.write(f"{confidence_color} **{finding['finding']}** (Confidence: {finding['confidence']:.2f})")
                st.write(f"   Location: {finding.get('location', 'N/A')}")
                st.write(f"   Severity: {finding.get('severity', 'N/A')}")
        else:
            st.success("✅ No significant findings detected")
    
    with col2:
        st.subheader("Quantitative Metrics")
        if results.get('quantitative_metrics'):
            for metric, value in results['quantitative_metrics'].items():
                st.write(f"**{metric}:** {value}")
        
        st.subheader("Overall Confidence")
        st.write(f"**Diagnostic Confidence:** {results.get('confidence', 0):.2f}")
        st.write(f"**Image Quality:** {results.get('image_quality', 'Good')}")

def display_findings_details():
    results = st.session_state.analysis_results
    
    st.subheader("Detailed Findings Analysis")
    
    if results.get('detailed_findings'):
        for category, findings in results['detailed_findings'].items():
            with st.expander(f"{category} ({len(findings)} findings)"):
                for finding in findings:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{finding['description']}**")
                    with col2:
                        st.write(f"Confidence: {finding['confidence']:.2f}")
                    with col3:
                        severity_color = {
                            'MILD': '🟢', 'MODERATE': '🟡', 'SEVERE': '🔴'
                        }.get(finding.get('severity', 'MILD'), '⚪')
                        st.write(f"{severity_color} {finding.get('severity', 'N/A')}")

def display_clinical_report():
    results = st.session_state.analysis_results
    
    st.subheader("Clinical Report")
    
    report = st.session_state.report_generator.generate_report(
        findings=results,
        patient_data=st.session_state.current_study['patient_info'],
        study_info={
            "modality": st.session_state.current_study['modality'],
            "body_part": st.session_state.current_study['anatomy']
        }
    )
    
    st.text_area("Radiology Report", report['full_report'], height=300)
    
    st.subheader("Clinical Recommendations")
    for recommendation in report.get('recommendations', []):
        st.write(f"• {recommendation}")

def display_quality_metrics():
    results = st.session_state.analysis_results
    
    st.subheader("Analysis Quality Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Diagnostic Confidence", f"{results.get('confidence', 0)*100:.1f}%")
        st.metric("Image Quality Score", f"{results.get('image_quality_score', 0)*100:.1f}%")
    
    with col2:
        st.metric("Findings Count", len(results.get('primary_findings', [])))
        st.metric("Segmentation Quality", f"{results.get('segmentation_quality', 0)*100:.1f}%")
    
    with col3:
        st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
        st.metric("Uncertainty Score", f"{results.get('uncertainty', 0)*100:.1f}%")

if __name__ == "__main__":
    main()