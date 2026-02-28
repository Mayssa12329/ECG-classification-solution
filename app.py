"""
EdgeCardio - Streamlit Demo Application
Interactive ECG classification with real-time inference
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import onnxruntime as ort
import time
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="EdgeCardio - ECG AI Diagnostics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'inference_history' not in st.session_state:
    st.session_state.inference_history = []


@st.cache_resource
def load_onnx_model(model_path='models/onnx/lightecgnet.onnx'):
    """Load ONNX model (cached)"""
    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        return session
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


@st.cache_data
def load_sample_ecgs(n_samples=10):
    """Load sample ECG signals"""
    sample_path = Path('data/sample_ecg.npy')
    
    if sample_path.exists():
        # Load real sample
        ecg = np.load(sample_path)
        samples = {'Real Sample': ecg}
        
        # Generate synthetic variations
        for i in range(n_samples - 1):
            noise = np.random.randn(*ecg.shape) * 0.05
            samples[f'Sample {i+1}'] = ecg + noise
        
        return samples
    else:
        # Generate synthetic ECG signals
        samples = {}
        for i in range(n_samples):
            samples[f'Sample {i+1}'] = generate_synthetic_ecg()
        return samples


def generate_synthetic_ecg(duration=10, fs=100, n_leads=12):
    """Generate synthetic ECG signal"""
    t = np.linspace(0, duration, duration * fs)
    ecg = np.zeros((len(t), n_leads))
    
    for lead in range(n_leads):
        # Heart rate: 60-100 bpm
        hr = np.random.uniform(60, 100)
        
        # P wave
        p_wave = 0.1 * np.sin(2 * np.pi * hr / 60 * t)
        
        # QRS complex
        qrs = 0.5 * np.sin(2 * np.pi * hr / 60 * t * 3) * np.exp(-((t % (60/hr) - 0.15)**2) / 0.001)
        
        # T wave
        t_wave = 0.2 * np.sin(2 * np.pi * hr / 60 * t - np.pi/4)
        
        # Combine + noise
        ecg[:, lead] = p_wave + qrs + t_wave + np.random.randn(len(t)) * 0.02
    
    return ecg


def plot_ecg_12lead(ecg_signal, sample_rate=100):
    """Plot 12-lead ECG using Plotly"""
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time_axis = np.arange(ecg_signal.shape[0]) / sample_rate
    
    # Create subplots
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, lead in enumerate(lead_names):
        offset = -i * 2  # Vertical offset for each lead
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=ecg_signal[:, i] + offset,
            name=lead,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=1.5),
            hovertemplate=f'<b>{lead}</b><br>Time: %{{x:.2f}}s<br>Amplitude: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="12-Lead ECG Signal",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', showticklabels=False)
    
    return fig


def plot_prediction_bars(probabilities):
    """Plot prediction probabilities as horizontal bar chart"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Color by probability
    colors = ['#2ecc71' if p > 0.5 else '#3498db' if p > 0.3 else '#95a5a6' for p in probs]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f'{p:.1%}' for p in probs],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Diagnostic Probabilities",
        xaxis_title="Probability",
        yaxis_title="",
        height=300,
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def plot_latency_history(history):
    """Plot inference latency over time"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['latency_ms'],
        mode='lines+markers',
        name='Latency',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))
    
    # Add mean line
    mean_lat = df['latency_ms'].mean()
    fig.add_hline(y=mean_lat, line_dash="dash", line_color="green", 
                  annotation_text=f"Mean: {mean_lat:.2f} ms")
    
    # Add target line (200ms)
    fig.add_hline(y=200, line_dash="dash", line_color="red", 
                  annotation_text="Target: 200 ms")
    
    fig.update_layout(
        title="Inference Latency Over Time",
        xaxis_title="Inference #",
        yaxis_title="Latency (ms)",
        height=300,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def run_inference(session, ecg_signal, demographics):
    """Run inference and measure latency"""
    # Prepare inputs
    ecg_input = ecg_signal.T[np.newaxis, :, :].astype(np.float32)  # (1, 12, 1000)
    demo_input = demographics[np.newaxis, :].astype(np.float32)     # (1, 6)
    
    # Run inference with timing
    start = time.perf_counter()
    outputs = session.run(
        None,
        {
            'ecg_signal': ecg_input,
            'demographics': demo_input
        }
    )
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    
    # Parse outputs
    logits = outputs[0][0]
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    class_names = {
        'NORM': 'Normal',
        'MI': 'Myocardial Infarction',
        'STTC': 'ST/T Change',
        'CD': 'Conduction Disturbance',
        'HYP': 'Hypertrophy'
    }
    
    predictions = {class_names[cls]: float(prob) for cls, prob in zip(classes, probs)}
    
    return predictions, latency_ms


def main():
    # Header
    st.markdown('<p class="main-header">🫀 EdgeCardio</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered ECG Diagnostics on Edge Devices</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=100)
        st.title("⚙️ Settings")
        
        # Model info
        st.subheader("📦 Model Information")
        model_path = Path('models/onnx/lightecgnet.onnx')
        
        if model_path.exists():
            model_size = model_path.stat().st_size / (1024**2)
            st.success(f"✅ Model loaded")
            st.metric("Model Size", f"{model_size:.2f} MB")
        else:
            st.error("❌ Model not found")
            st.info("Run `python main.py --use-kaggle --epochs 50` first")
            return
        
        st.divider()
        
        # Demographics inputs
        st.subheader("👤 Patient Demographics")
        
        age = st.slider("Age", 0, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        weight = st.slider("Weight (kg)", 30, 150, 75)
        
        # Convert to model inputs
        sex_encoded = 1.0 if sex == "Male" else 0.0
        nurse = 0.0  # Default
        site = 0.0   # Default
        device = 0.0 # Default
        
        demographics = np.array([age, sex_encoded, weight, nurse, site, device], dtype=np.float32)
        
        st.divider()
        
        # Performance settings
        st.subheader("⚡ Performance")
        n_warmup = st.number_input("Warmup iterations", 1, 50, 5)
        show_latency_history = st.checkbox("Show latency history", value=True)
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.info(
            "**EdgeCardio** is a lightweight CNN for ECG classification, "
            "optimized for edge devices like Raspberry Pi 5.\n\n"
            "**Classes:**\n"
            "- Normal (NORM)\n"
            "- Myocardial Infarction (MI)\n"
            "- ST/T Change (STTC)\n"
            "- Conduction Disturbance (CD)\n"
            "- Hypertrophy (HYP)"
        )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Live Inference", "📈 Benchmark", "📋 Model Info"])
    
    # === TAB 1: Live Inference ===
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📡 ECG Signal Selection")
            
            # Load samples
            samples = load_sample_ecgs(n_samples=5)
            selected_sample = st.selectbox("Select ECG sample", list(samples.keys()))
            ecg_signal = samples[selected_sample]
            
            # Plot ECG
            fig_ecg = plot_ecg_12lead(ecg_signal)
            st.plotly_chart(fig_ecg, use_container_width=True)
        
        with col2:
            st.subheader("🔬 Run Inference")
            
            if st.button("🚀 Analyze ECG", type="primary", use_container_width=True):
                with st.spinner("Running inference..."):
                    # Load model
                    session = load_onnx_model()
                    
                    if session is None:
                        st.error("Failed to load model")
                        return
                    
                    # Warmup
                    for _ in range(n_warmup):
                        _ = run_inference(session, ecg_signal, demographics)
                    
                    # Inference
                    predictions, latency_ms = run_inference(session, ecg_signal, demographics)
                    
                    # Store in history
                    st.session_state.inference_history.append({
                        'latency_ms': latency_ms,
                        'timestamp': time.time()
                    })
                    
                    # Display results
                    st.success(f"✅ Inference completed in **{latency_ms:.2f} ms**")
                    
                    # Primary diagnosis
                    primary_diagnosis = max(predictions, key=predictions.get)
                    confidence = predictions[primary_diagnosis]
                    
                    st.metric(
                        label="Primary Diagnosis",
                        value=primary_diagnosis,
                        delta=f"Confidence: {confidence:.1%}"
                    )
                    
                    # Plot probabilities
                    fig_probs = plot_prediction_bars(predictions)
                    st.plotly_chart(fig_probs, use_container_width=True)
                    
                    # Detailed probabilities
                    st.subheader("📊 Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        'Condition': list(predictions.keys()),
                        'Probability': [f"{v:.2%}" for v in predictions.values()]
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # Latency history
        if show_latency_history and st.session_state.inference_history:
            st.divider()
            st.subheader("⏱️ Performance History")
            fig_latency = plot_latency_history(st.session_state.inference_history)
            if fig_latency:
                st.plotly_chart(fig_latency, use_container_width=True)
            
            # Stats
            latencies = [h['latency_ms'] for h in st.session_state.inference_history]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Latency", f"{np.mean(latencies):.2f} ms")
            col2.metric("Median Latency", f"{np.median(latencies):.2f} ms")
            col3.metric("Min Latency", f"{np.min(latencies):.2f} ms")
            col4.metric("Max Latency", f"{np.max(latencies):.2f} ms")
    
    # === TAB 2: Benchmark ===
    with tab2:
        st.subheader("⚡ Performance Benchmark")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_runs = st.number_input("Number of benchmark runs", 10, 1000, 100)
        
        with col2:
            st.write("")  # Spacing
        
        if st.button("🏁 Run Benchmark", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            session = load_onnx_model()
            if session is None:
                st.error("Failed to load model")
                return
            
            # Get random sample
            samples = load_sample_ecgs(1)
            ecg_signal = list(samples.values())[0]
            
            # Warmup
            status_text.text("Warming up...")
            for _ in range(10):
                _ = run_inference(session, ecg_signal, demographics)
            
            # Benchmark
            latencies = []
            for i in range(n_runs):
                _, lat = run_inference(session, ecg_signal, demographics)
                latencies.append(lat)
                
                progress_bar.progress((i + 1) / n_runs)
                if (i + 1) % 20 == 0:
                    status_text.text(f"Running benchmark: {i+1}/{n_runs}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Results
            latencies = np.array(latencies)
            
            st.success("✅ Benchmark completed!")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{latencies.mean():.2f} ms")
            col2.metric("Median", f"{np.median(latencies):.2f} ms")
            col3.metric("P95", f"{np.percentile(latencies, 95):.2f} ms")
            col4.metric("P99", f"{np.percentile(latencies, 99):.2f} ms")
            
            # Histogram
            fig_hist = go.Figure(go.Histogram(
                x=latencies,
                nbinsx=30,
                marker_color='#3498db',
                name='Latency Distribution'
            ))
            
            fig_hist.add_vline(x=latencies.mean(), line_dash="dash", line_color="green",
                              annotation_text=f"Mean: {latencies.mean():.2f} ms")
            fig_hist.add_vline(x=200, line_dash="dash", line_color="red",
                              annotation_text="Target: 200 ms")
            
            fig_hist.update_layout(
                title="Latency Distribution",
                xaxis_title="Latency (ms)",
                yaxis_title="Count",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Throughput
            throughput = 1000.0 / latencies.mean()
            st.metric("Throughput", f"{throughput:.1f} samples/second")
            
            # Constraint check
            if latencies.mean() < 200:
                st.success(f"✅ Latency constraint satisfied ({latencies.mean():.2f} ms < 200 ms)")
            else:
                st.error(f"❌ Latency constraint violated ({latencies.mean():.2f} ms > 200 ms)")
    
    # === TAB 3: Model Info ===
    with tab3:
        st.subheader("🔍 Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **LightECGNet Architecture:**
            - Input: 12-lead ECG (12 × 1000) + Demographics (6 features)
            - Multi-scale temporal convolutions
            - Squeeze-and-Excitation blocks
            - Depthwise separable convolutions
            - Global Average Pooling
            - Output: 5 diagnostic classes
            
            **Optimizations:**
            - Depthwise separable convolutions (8× fewer params)
            - Channel reduction (64 base channels)
            - ONNX quantization-ready
            - CPU-optimized inference
            """)
        
        with col2:
            # Model stats
            model_path = Path('models/onnx/lightecgnet.onnx')
            if model_path.exists():
                model_size = model_path.stat().st_size / (1024**2)
                
                st.metric("Model Size", f"{model_size:.2f} MB")
                st.metric("Target Device", "Raspberry Pi 5")
                st.metric("Framework", "ONNX Runtime")
                st.metric("Precision", "FP32")
                
                # Constraints
                st.divider()
                st.markdown("**PhysioNet Challenge Constraints:**")
                
                if model_size < 100:
                    st.success(f"✅ Size < 100 MB ({model_size:.2f} MB)")
                else:
                    st.error(f"❌ Size > 100 MB ({model_size:.2f} MB)")
                
                # Load benchmark results if available
                results_path = Path('results/benchmark_results.json')
                if results_path.exists():
                    with open(results_path) as f:
                        bench_results = json.load(f)
                    
                    mean_lat = bench_results.get('mean_latency_ms', 0)
                    if mean_lat < 200:
                        st.success(f"✅ Latency < 200 ms ({mean_lat:.2f} ms)")
                    else:
                        st.error(f"❌ Latency > 200 ms ({mean_lat:.2f} ms)")


if __name__ == "__main__":
    main()