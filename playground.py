import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import torch
import psutil
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import shutil
import zipfile
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, f1_score

# --------------------------
# 0. Configuration
# --------------------------
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = {}

# --------------------------
# 1. Model Upload & Configuration (Fixed)
# --------------------------
def load_model(uploaded_model):
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_models"
        os.makedirs(temp_dir, exist_ok=True)
        
        if uploaded_model.name.endswith('.pt'):
            # Save YOLO model to temp file
            temp_path = os.path.join(temp_dir, uploaded_model.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            return ('yolo', YOLO(temp_path))
        
        elif uploaded_model.name.endswith('.pth'):
            # Load PyTorch model from buffer
            buffer = BytesIO(uploaded_model.read())
            return ('pytorch', torch.load(buffer))
        
        elif uploaded_model.name.endswith('.pkl'):
            # Load sklearn model from buffer
            buffer = BytesIO(uploaded_model.read())
            return ('sklearn', joblib.load(buffer))
    
    except Exception as e:
        st.error(f"Error loading {uploaded_model.name}: {str(e)}")
    return (None, None)

def cleanup_temp_files():
    """Safely remove temporary model files"""
    temp_dir = "temp_models"
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        st.error(f"Error cleaning temporary files: {str(e)}")
    finally:
        # Ensure directory exists for next session
        os.makedirs(temp_dir, exist_ok=True)

def handle_model_upload():
    st.sidebar.header("1. Upload Models & Configs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload models/configs (.pt, .pkl, .pth, .yaml, .yml)",
        type=["pt", "pkl", "pth", "yaml", "yml"],
        accept_multiple_files=True
    )
    
    for file in uploaded_files[:3]:  # Limit to 3 models
        if file.name.endswith(('.yaml', '.yml')):
            try:
                st.session_state.dataset_config = yaml.safe_load(file)
                st.success(f"Loaded config: {list(st.session_state.dataset_config.keys())}")
            except Exception as e:
                st.error(f"Invalid YAML: {str(e)}")
        else:
            model_type, model = load_model(file)
            if model:
                model_name = f"{file.name} ({model_type})"
                st.session_state.models[model_name] = (model_type, model)
                st.success(f"Loaded {model_name}")

# --------------------------
# 2. Data Upload & Processing
# --------------------------
def process_zip(zip_file):
    """Extract images and labels from ZIP while preserving structure"""
    images = []
    labels = {}
    
    with zipfile.ZipFile(zip_file) as zf:
        for fileinfo in zf.infolist():
            if fileinfo.is_dir():
                continue
                
            filename = fileinfo.filename
            base = os.path.splitext(os.path.basename(filename))[0]
            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                with zf.open(filename) as f:
                    # Read image data into memory buffer
                    img_buffer = BytesIO(f.read())
                    img = Image.open(img_buffer)
                    img.filename = filename  # Preserve original filename
                    images.append((base, img))
                    
            elif filename.lower().endswith(('.yaml', '.yml', '.txt')):
                with zf.open(filename) as f:
                    labels[base] = f.read().decode()
    
    return images, labels

def handle_data_upload():
    st.sidebar.header("2. Upload Test Data")
    
    # For traditional models
    data_file = st.sidebar.file_uploader("Tabular Data (CSV)", type=["csv"])
    
    # For computer vision models
    dataset_zip = st.sidebar.file_uploader(
        "Dataset (ZIP with images + labels)",
        type=["zip"],
        help="ZIP file containing images and corresponding YAML/TXT labels"
    )
    
    return data_file, dataset_zip

# --------------------------
# 3. Model Inference (Fixed)
# --------------------------
def match_labels(images, labels):
    """Match images with corresponding labels using base filename"""
    return [(img, labels.get(base)) for base, img in images]

def get_system_stats():
    """Collect system performance metrics"""
    stats = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'gpu_usage': 0,
        'gpu_memory': 0
    }
    
    if torch.cuda.is_available():
        stats['gpu_usage'] = torch.cuda.utilization()
        stats['gpu_memory'] = torch.cuda.memory_allocated() / (1024 ** 3)
    
    return stats

def run_inference():
    st.session_state.metrics = {}
    st.session_state.system_stats = {}
    data_file, dataset_zip = handle_data_upload()

    # Common configuration
    batch_size = st.sidebar.slider("Batch size", 1, 128, 32)
    sample_count = 0

    # Process tabular data (CSV)
    if data_file:
        df = pd.read_csv(data_file)
        if 'target' in df.columns:
            y = df['target']
            X = df.drop(columns=['target'])
            sample_count = len(df)

    # Process image data (ZIP)
    if dataset_zip:
        images, _ = process_zip(dataset_zip)
        sample_count = len(images)

    if sample_count == 0:
        st.warning("No data loaded!")
        return

    for model_name, (model_type, model) in st.session_state.models.items():
        start_time = time.time()
        stats_start = get_system_stats()
        metrics = {
            'total_samples': sample_count,
            'batch_size': batch_size,
            'time_per_sample': None,
            'batch_speed': None,
            'cpu_usage': None,
            'gpu_usage': None,
            'memory_usage': None
        }

        try:
            if model_type == 'sklearn' and data_file is not None:
                # Scikit-learn batch processing
                y_pred = []
                for i in range(0, len(X), batch_size):
                    batch = X.iloc[i:i+batch_size]
                    y_pred.extend(model.predict(batch))
                
                metrics.update({
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted'),
                    'f1': f1_score(y, y_pred, average='weighted')
                })

            elif model_type == 'yolo' and dataset_zip is not None:
                # YOLO batch processing
                total_detections = 0
                for i in range(0, len(images), batch_size):
                    batch = images[i:i+batch_size]
                    batch_images = [np.array(img) for _, img in batch]
                    
                    # Warmup
                    if i == 0:
                        model(batch_images[0])
                    
                    # Batch inference
                    results = model(batch_images, verbose=False)
                    total_detections += sum(len(r.boxes) for r in results)
                
                metrics['avg_detections'] = total_detections / sample_count

            # Common timing metrics
            total_time = time.time() - start_time
            metrics.update({
                'time_per_sample': total_time / sample_count,
                'batch_speed': sample_count / total_time,
            })

            # System metrics
            stats_end = get_system_stats()
            st.session_state.system_stats[model_name] = {
                'cpu': np.mean([stats_start['cpu_usage'], stats_end['cpu_usage']]),
                'gpu': np.mean([stats_start['gpu_usage'], stats_end['gpu_usage']]),
                'memory': np.mean([stats_start['memory_usage'], stats_end['memory_usage']])
            }

            # Store metrics
            st.session_state.metrics[model_name] = metrics

        except Exception as e:
            st.error(f"Error with {model_name}: {str(e)}")

    # Add system metrics to main metrics
    for model_name, metrics in st.session_state.metrics.items():
        sys_metrics = st.session_state.system_stats.get(model_name, {})
        metrics.update({
            'cpu_usage (%)': sys_metrics.get('cpu', 0),
            'gpu_usage (%)': sys_metrics.get('gpu', 0),
            'memory_usage (%)': sys_metrics.get('memory', 0)
        })
# --------------------------
# 4. System Metrics Dashboard
# --------------------------
def show_dashboard():
    if not st.session_state.metrics:
        st.warning("⚠️ No metrics available. Run inference first!")
        return
    
    st.title("📊 Model Performance Dashboard")
    
    # Real-time System Monitor
    st.header("Live System Metrics")
    sys_cols = st.columns(4)
    with sys_cols[0]:
        st.metric("CPU Usage (%)", psutil.cpu_percent())
    with sys_cols[1]:
        st.metric("Memory Usage (%)", psutil.virtual_memory().percent)
    with sys_cols[2]:
        gpu_usage = torch.cuda.utilization() if torch.cuda.is_available() else 0
        st.metric("GPU Usage (%)", gpu_usage)
    with sys_cols[3]:
        gpu_mem = torch.cuda.memory_allocated()/(1024**3) if torch.cuda.is_available() else 0
        st.metric("GPU Memory (GB)", f"{gpu_mem:.2f}")

    # Performance Tabs
    tab1, tab2, tab3 = st.tabs(["Performance Matrix", "Timing Analysis", "Hardware Utilization"])

    with tab1:
        st.subheader("Performance Matrix")
        metrics_df = pd.DataFrame(st.session_state.metrics).T
        
        # Define desired columns in priority order
        possible_columns = [
            'time_per_sample', 'batch_speed', 'cpu_usage (%)',
            'gpu_usage (%)', 'memory_usage (%)', 'accuracy',
            'precision', 'f1', 'avg_detections'
        ]
        
        # Filter to only existing columns
        existing_columns = [col for col in possible_columns if col in metrics_df.columns]
        filtered_df = metrics_df[existing_columns]
        
        # Formatting
        fmt_rules = {
            'time_per_sample': '{:.4f} s',
            'batch_speed': '{:.1f} samples/s',
            'cpu_usage (%)': '{:.1f}%',
            'gpu_usage (%)': '{:.1f}%',
            'memory_usage (%)': '{:.1f}%',
            'accuracy': '{:.2%}',
            'precision': '{:.2%}',
            'f1': '{:.2%}',
            'avg_detections': '{:.1f}'
        }

        for col, fmt in fmt_rules.items():
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else x)
        
        st.dataframe(
            filtered_df.style.background_gradient(cmap="Blues"),
            use_container_width=True
        )

    
    with tab2:
        st.markdown("""
            <h2 style='font-size:28px; border-bottom:2px solid #4F8BF9; padding-bottom:5px;'>
                ⏱️ Comparative Timing Analysis
            </h2>
        """, unsafe_allow_html=True)

        if not st.session_state.metrics:
            st.warning("No timing data available")
            return

        # Create time metrics dataframe
        timing_data = pd.DataFrame(st.session_state.metrics).T
        timing_data = timing_data[['time_per_sample', 'batch_speed']]
        timing_data.columns = ['Time per Sample (s)', 'Batch Speed (samples/s)']

        # Comparative bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set positions and width for bars
        bar_width = 0.2
        index = np.arange(len(timing_data))
        
        # Plot time per sample
        bars1 = ax.bar(
            index - bar_width/2,
            timing_data['Time per Sample (s)'],
            bar_width,
            color='#FF6F61',
            label='Time per Sample'
        )
        
        # Plot batch speed on secondary axis
        ax2 = ax.twinx()
        bars2 = ax2.bar(
            index + bar_width/2,
            timing_data['Batch Speed (samples/s)'],
            bar_width,
            color='#6B5B95',
            label='Batch Speed'
        )

        # Formatting
        ax.set_title('Model Performance Comparison', fontsize=16)
        ax.set_xticks(index)
        ax.set_xticklabels(timing_data.index, rotation=45, ha='right')
        ax.set_ylabel('Time per Sample (seconds)', color='#FF6F61')
        ax2.set_ylabel('Batch Speed (samples/second)', color='#6B5B95')
        ax.tick_params(axis='y', colors='#FF6F61')
        ax2.tick_params(axis='y', colors='#6B5B95')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### 📈 Performance Statistics")
        
        # Get statistics and select columns correctly
        stats_df = timing_data.describe().T[['mean', 'min', 'max']]
        stats_df.columns = ['Average', 'Minimum', 'Maximum']
        
        # Formatting
        formatted_stats = stats_df.style.format({
            'Average': '{:.4f} s',
            'Minimum': '{:.4f} s',
            'Maximum': '{:.4f} s'
        }).background_gradient(cmap='YlGnBu')

        st.dataframe(
            formatted_stats,
            use_container_width=True,
            height=150
        )
        # Throughput analysis
        st.markdown("#### 📊 Throughput Breakdown")
        cols = st.columns(3)
        
        with cols[0]:
            avg_throughput = timing_data['Batch Speed (samples/s)'].mean()
            st.metric("Average Throughput", f"{avg_throughput:.1f} samples/s")
            
        with cols[1]:
            max_throughput = timing_data['Batch Speed (samples/s)'].max()
            st.metric("Peak Throughput", f"{max_throughput:.1f} samples/s")
            
        with cols[2]:
            efficiency = (timing_data['Batch Speed (samples/s)'] / 
                         timing_data['Time per Sample (s)']).mean()
            st.metric("Processing Efficiency", f"{efficiency:.2f}x")

        # Model comparison table
        st.markdown("#### 📋 Model-wise Performance")
        comparison_df = timing_data.copy()
        comparison_df['Relative Speed'] = (comparison_df['Batch Speed (samples/s)'] / 
                                         comparison_df['Batch Speed (samples/s)'].max())
        comparison_df['Time Variance'] = comparison_df['Time per Sample (s)'] / comparison_df['Time per Sample (s)'].mean()
        
        st.dataframe(
            comparison_df.style.format({
                'Time per Sample (s)': '{:.4f} s',
                'Batch Speed (samples/s)': '{:.1f}',
                'Relative Speed': '{:.2%}',
                'Time Variance': '{:.2f}x'
            }).bar(subset=['Relative Speed'], color='#000000'),
            use_container_width=True
        )
    with tab3:
        st.subheader("Hardware Utilization Analytics")
        
        if not st.session_state.system_stats:
            st.warning("No hardware metrics collected")
            return

        # Convert to DataFrame
        stats_df = pd.DataFrame(st.session_state.system_stats).T
        stats_df = stats_df.rename(columns={
            'cpu': 'CPU Usage (%)',
            'gpu': 'GPU Usage (%)',
            'memory': 'Memory Usage (%)'
        })

        # Add summary statistics
        summary_stats = pd.DataFrame({
            'Average': stats_df.mean(),
            'Peak': stats_df.max(),
            'Minimum': stats_df.min(),
            'Std Dev': stats_df.std()
        }).T

        # System-wide metrics
        st.markdown("**System-wide Summary**")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Avg CPU Utilization", f"{summary_stats.loc['Average', 'CPU Usage (%)']:.1f}%")
        with cols[1]:
            st.metric("Peak GPU Load", f"{summary_stats.loc['Peak', 'GPU Usage (%)']:.1f}%")
        with cols[2]:
            st.metric("Max Memory Pressure", f"{summary_stats.loc['Peak', 'Memory Usage (%)']:.1f}%")
        with cols[3]:
            st.metric("Performance Variance", f"{summary_stats.loc['Std Dev'].mean():.1f}%")

        # Detailed model breakdown
        st.markdown("**Per-Model Resource Consumption**")
        
        # Format numerical values
        formatted_df = stats_df.style.format({
            'CPU Usage (%)': '{:.1f}%',
            'GPU Usage (%)': '{:.1f}%',
            'Memory Usage (%)': '{:.1f}%'
        }).background_gradient(subset=['CPU Usage (%)'], cmap='Reds')\
          .background_gradient(subset=['GPU Usage (%)'], cmap='Oranges')\
          .background_gradient(subset=['Memory Usage (%)'], cmap='Blues')

        st.dataframe(
            formatted_df,
            use_container_width=True,
            height=min(400, 35 * len(stats_df))
        )
        # Efficiency metrics
        st.markdown("**Resource Efficiency Scores**")
        efficiency_df = pd.DataFrame({
            'CPU Efficiency': (stats_df['CPU Usage (%)'] / 100) * (1/stats_df.index.map(
                lambda x: st.session_state.metrics[x]['time_per_sample'])),
            'GPU Efficiency': (stats_df['GPU Usage (%)'] / 100) * (1/stats_df.index.map(
                lambda x: st.session_state.metrics[x]['time_per_sample']))
        }).fillna(0)

        st.write("""
            Efficiency scores combine resource utilization with processing speed:
            - Higher values indicate better resource-time efficiency
            - Scores normalized across models
        """)
        
        st.dataframe(
            efficiency_df.style.format('{:.2f}').background_gradient(cmap='viridis'),
            use_container_width=True
        )

if __name__ == "__main__":
    try:
        st.title("🧪 Model Testing Playground")
        handle_model_upload()
        run_inference()
        show_dashboard()
    finally:
        cleanup_temp_files()