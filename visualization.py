import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import Dict, Any, List, Optional

class MedicalVisualizer:
    def __init__(self):
        self.colors = {
            'normal': '#2E86AB',
            'abnormal': '#A23B72',
            'critical': '#F18F01',
            'background': '#F7F7F7'
        }
    
    def create_findings_plot(self, findings: Dict[str, Any]) -> go.Figure:
        primary_findings = findings.get('primary_findings', [])
        
        if not primary_findings:
            return self._create_no_findings_plot()
        
        findings_names = [f['finding'] for f in primary_findings]
        confidences = [f['confidence'] for f in primary_findings]
        severities = [f.get('severity', 'MILD') for f in primary_findings]
        
        colors = []
        for severity in severities:
            if severity == 'SEVERE':
                colors.append(self.colors['critical'])
            elif severity == 'MODERATE':
                colors.append(self.colors['abnormal'])
            else:
                colors.append(self.colors['normal'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=findings_names,
                y=confidences,
                marker_color=colors,
                text=[f'{c:.2f}' for c in confidences],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Diagnostic Findings Analysis',
            xaxis_title='Findings',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_confidence_radar(self, findings: Dict[str, Any]) -> go.Figure:
        metrics = findings.get('quantitative_metrics', {})
        
        if not metrics:
            categories = ['Detection', 'Segmentation', 'Classification']
            values = [0.7, 0.6, 0.8]
        else:
            categories = list(metrics.keys())[:6]
            values = list(metrics.values())[:6]
            values = [float(v) for v in values]
        
        fig = go.Figure(data=
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(46, 134, 171, 0.3)',
                line=dict(color=self.colors['normal']),
                name='Performance Metrics'
            )
        )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title='AI Analysis Confidence Radar',
            height=400
        )
        
        return fig
    
    def create_segmentation_overlay(self, image: np.ndarray, mask: np.ndarray, 
                                  finding_name: str) -> go.Figure:
        
        if len(image.shape) == 3:
            slice_idx = image.shape[0] // 2
            image_slice = image[slice_idx]
            mask_slice = mask[slice_idx]
        else:
            image_slice = image
            mask_slice = mask
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=['Original Image', 'Segmentation Overlay'])
        
        fig.add_trace(
            go.Heatmap(z=image_slice, colorscale='gray', showscale=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(z=image_slice, colorscale='gray', showscale=False),
            row=1, col=2
        )
        
        overlay_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
        
        fig.add_trace(
            go.Heatmap(
                z=overlay_mask,
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(162, 59, 114, 0.6)']],
                showscale=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Segmentation Results: {finding_name}',
            height=400
        )
        
        return fig
    
    def create_risk_assessment_chart(self, risk_assessment: Dict[str, Any]) -> go.Figure:
        risk_level = risk_assessment.get('risk_level', 'LOW')
        risk_score = risk_assessment.get('risk_score', 0)
        critical_count = risk_assessment.get('critical_findings_count', 0)
        
        risk_colors = {
            'LOW': '#2E86AB',
            'MODERATE': '#F18F01', 
            'HIGH': '#A23B72'
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Risk Score"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 10]},
                'bar': {'color': risk_colors.get(risk_level, '#2E86AB')},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 6], 'color': "gray"},
                    {'range': [6, 10], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6
                }
            }
        ))
        
        fig.update_layout(
            title='Clinical Risk Assessment',
            height=300
        )
        
        return fig
    
    def create_timeline_visualization(self, previous_studies: List[Dict[str, Any]], 
                                    current_study: Dict[str, Any]) -> go.Figure:
        
        if not previous_studies:
            return self._create_single_study_plot(current_study)
        
        dates = [study.get('date', f'Study_{i}') for i, study in enumerate(previous_studies)]
        dates.append('Current')
        
        risk_scores = [study.get('risk_score', 0) for study in previous_studies]
        risk_scores.append(current_study.get('risk_score', 0))
        
        finding_counts = [len(study.get('primary_findings', [])) for study in previous_studies]
        finding_counts.append(len(current_study.get('primary_findings', [])))
        
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=['Risk Score Trend', 'Finding Count Trend'])
        
        fig.add_trace(
            go.Scatter(x=dates, y=risk_scores, mode='lines+markers', 
                      name='Risk Score', line=dict(color=self.colors['critical'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=dates, y=finding_counts, name='Finding Count',
                  marker_color=self.colors['abnormal']),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Longitudinal Analysis',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def _create_no_findings_plot(self) -> go.Figure:
        fig = go.Figure()
        
        fig.add_annotation(
            text="No Significant Findings Detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.colors['normal'])
        )
        
        fig.update_layout(
            title='Diagnostic Findings Analysis',
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=400,
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def _create_single_study_plot(self, study: Dict[str, Any]) -> go.Figure:
        fig = go.Figure()
        
        confidence = study.get('confidence', 0)
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            title = {'text': "Diagnostic Confidence"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': self.colors['normal']},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(
            title='Single Study Analysis',
            height=300
        )
        
        return fig

class AdvancedMedicalVisualizer(MedicalVisualizer):
    def __init__(self):
        super().__init__()
    
    def create_3d_volume_rendering(self, volume: np.ndarray, masks: Dict[str, np.ndarray]) -> go.Figure:
        if len(volume.shape) != 3:
            raise ValueError("Input must be a 3D volume")
        
        x, y, z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
        
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=volume.mean() - volume.std(),
            isomax=volume.mean() + volume.std(),
            opacity=0.1,
            surface_count=17,
            colorscale='Gray'
        ))
        
        for mask_name, mask_data in masks.items():
            if mask_data.shape == volume.shape:
                mask_points = np.where(mask_data > 0)
                
                if len(mask_points[0]) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=mask_points[0],
                        y=mask_points[1],
                        z=mask_points[2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=self.colors['abnormal'],
                            opacity=0.6
                        ),
                        name=mask_name
                    ))
        
        fig.update_layout(
            title='3D Volume Rendering with Segmentation',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            height=600
        )
        
        return fig
    
    def create_comparative_analysis(self, study_a: Dict[str, Any], 
                                  study_b: Dict[str, Any]) -> go.Figure:
        
        findings_a = study_a.get('primary_findings', [])
        findings_b = study_b.get('primary_findings', [])
        
        common_findings = set([f['finding'] for f in findings_a]) & set([f['finding'] for f in findings_b])
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=['Study A', 'Study B'],
                          specs=[[{"type": "pie"}, {"type": "pie"}]])
        
        if findings_a:
            labels_a = [f['finding'] for f in findings_a]
            values_a = [f['confidence'] for f in findings_a]
            fig.add_trace(go.Pie(labels=labels_a, values=values_a, name="Study A"), 1, 1)
        
        if findings_b:
            labels_b = [f['finding'] for f in findings_b]
            values_b = [f['confidence'] for f in findings_b]
            fig.add_trace(go.Pie(labels=labels_b, values=values_b, name="Study B"), 1, 2)
        
        fig.update_layout(
            title='Comparative Study Analysis',
            height=400
        )
        
        return fig