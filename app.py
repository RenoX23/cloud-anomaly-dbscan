"""
Cloud Anomaly Detection with DBSCAN
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add paths
sys.path.append('data')
sys.path.append('clustering')
sys.path.append('visualization')

from generate_metrics import generate_cloud_metrics
from dbscan_cluster import CloudAnomalyDetector, find_optimal_epsilon

# Page config
st.set_page_config(
    page_title="Cloud Anomaly Detection - DBSCAN",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">☁️ Cloud Resource Anomaly Detection with DBSCAN</div>', unsafe_allow_html=True)
st.markdown("**Density-Based Clustering for Infrastructure Monitoring** | M.Tech AI/ML Project")
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png", width=200)
    st.markdown("## 🎛️ Configuration")

    st.markdown("### Data Generation")
    n_points = st.slider("Number of data points", 500, 5000, 2000, 100)
    n_servers = st.slider("Number of servers", 3, 10, 5)
    anomaly_rate = st.slider("Anomaly rate (%)", 1, 15, 5) / 100

    st.markdown("### DBSCAN Parameters")

    auto_tune = st.checkbox("Auto-tune epsilon", value=True)

    if auto_tune:
        st.info("Epsilon will be automatically calculated using k-distance method")
        eps = None
    else:
        eps = st.slider("Epsilon (ε)", 0.1, 2.0, 0.5, 0.1)

    min_samples = st.slider("Min samples", 3, 20, 5)

    st.markdown("---")

    generate_button = st.button("🚀 Generate & Analyze", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard demonstrates **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)
    for detecting anomalies in cloud infrastructure metrics.

    **Key Features:**
    - Real-time parameter tuning
    - Interactive visualizations
    - Performance metrics
    - Anomaly analysis
    """)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

# Main content
if generate_button or st.session_state.data_generated:

    if generate_button:
        with st.spinner("Generating synthetic cloud metrics..."):
            # Generate data
            df = generate_cloud_metrics(n_points=n_points, n_servers=n_servers, anomaly_rate=anomaly_rate)
            st.session_state.df = df

            # Auto-tune epsilon if needed
            if auto_tune:
                suggested_eps, distances = find_optimal_epsilon(df, min_samples=min_samples)
                eps = suggested_eps
                st.session_state.eps = eps
            else:
                st.session_state.eps = eps

            # Apply DBSCAN
            detector = CloudAnomalyDetector(eps=eps, min_samples=min_samples)
            labels = detector.fit_predict(df)
            st.session_state.labels = labels

            # Store analysis
            analysis = detector.analyze_clusters(df, labels)
            st.session_state.analysis = analysis

            # Add labels to dataframe
            df['cluster'] = labels
            df['detected_anomaly'] = (labels == -1)
            st.session_state.df = df

            st.session_state.data_generated = True
            st.session_state.detector = detector

    # Retrieve from session state
    df = st.session_state.df
    labels = st.session_state.labels
    analysis = st.session_state.analysis
    eps = st.session_state.eps

    # === SECTION 1: KEY METRICS ===
    st.markdown('<div class="sub-header">📊 Key Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Data Points", f"{len(df):,}")

    with col2:
        st.metric("Clusters Found", analysis['n_clusters'])

    with col3:
        st.metric("Anomalies Detected", f"{analysis['n_anomalies']}")

    with col4:
        st.metric("Anomaly Rate", f"{analysis['anomaly_percentage']:.2f}%")

    with col5:
        st.metric("Epsilon (ε)", f"{eps:.3f}")

    st.divider()

    # === SECTION 2: PERFORMANCE EVALUATION ===
    st.markdown('<div class="sub-header">🎯 Performance Evaluation</div>', unsafe_allow_html=True)

    # Calculate metrics
    actual = df['is_anomaly'].astype(int)
    detected = (labels == -1).astype(int)

    cm = confusion_matrix(actual, detected)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")

    with col2:
        st.metric("Precision", f"{precision:.3f}")

    with col3:
        st.metric("Recall", f"{recall:.3f}")

    with col4:
        st.metric("F1-Score", f"{f1:.3f}")

    # Confusion Matrix
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax.set_xlabel('Detected')
        ax.set_ylabel('Actual')
        ax.set_title('Actual vs Detected')
        st.pyplot(fig_cm)

    with col2:
        st.markdown("#### Performance Interpretation")
        st.markdown(f"""
        **Results Summary:**
        - **True Positives (TP)**: {tp} - Correctly identified anomalies ✅
        - **True Negatives (TN)**: {tn} - Correctly identified normal behavior ✅
        - **False Positives (FP)**: {fp} - Normal flagged as anomaly ⚠️
        - **False Negatives (FN)**: {fn} - Missed anomalies ❌

        **Interpretation:**
        - **Precision ({precision:.3f})**: Of all points flagged as anomalies, {precision*100:.1f}% were actually anomalies
        - **Recall ({recall:.3f})**: Of all actual anomalies, we detected {recall*100:.1f}%
        - **F1-Score ({f1:.3f})**: Harmonic mean of precision and recall
        """)

    st.divider()

    # === SECTION 3: VISUALIZATIONS ===
    st.markdown('<div class="sub-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["2D Clusters", "3D Clusters", "Timeline Analysis", "Parameter Sensitivity"])

    with tab1:
        st.markdown("#### 2D Cluster Visualization (CPU vs Memory)")

        # Create plotly figure
        df_plot = df.copy()
        df_plot['Cluster'] = df_plot['cluster'].apply(lambda x: 'Anomaly' if x == -1 else f'Cluster {x}')

        fig = px.scatter(
            df_plot,
            x='cpu',
            y='memory',
            color='Cluster',
            color_discrete_map={'Anomaly': 'red'},
            title='DBSCAN Clustering: CPU vs Memory',
            labels={'cpu': 'CPU (%)', 'memory': 'Memory (%)'},
            hover_data=['server_id', 'network_io'],
            height=600
        )

        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGray')))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 3D Cluster Visualization (CPU, Memory, Network I/O)")

        fig_3d = px.scatter_3d(
            df_plot,
            x='cpu',
            y='memory',
            z='network_io',
            color='Cluster',
            color_discrete_map={'Anomaly': 'red'},
            title='3D DBSCAN Clustering',
            labels={'cpu': 'CPU (%)', 'memory': 'Memory (%)', 'network_io': 'Network I/O (%)'},
            hover_data=['server_id'],
            height=700
        )

        fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color='DarkSlateGray')))

        st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        st.markdown("#### Resource Metrics Timeline with Anomalies")

        # Create subplot with 3 charts
        fig_timeline = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU Usage Over Time', 'Memory Usage Over Time', 'Network I/O Over Time'),
            vertical_spacing=0.1
        )

        # Add normal and anomaly points
        normal_mask = df['detected_anomaly'] == False
        anomaly_mask = df['detected_anomaly'] == True

        # CPU
        fig_timeline.add_trace(
            go.Scatter(x=df[normal_mask].index, y=df[normal_mask]['cpu'],
                      mode='markers', name='Normal', marker=dict(size=4, color='blue', opacity=0.5)),
            row=1, col=1
        )
        fig_timeline.add_trace(
            go.Scatter(x=df[anomaly_mask].index, y=df[anomaly_mask]['cpu'],
                      mode='markers', name='Anomaly', marker=dict(size=8, color='red', symbol='x')),
            row=1, col=1
        )

        # Memory
        fig_timeline.add_trace(
            go.Scatter(x=df[normal_mask].index, y=df[normal_mask]['memory'],
                      mode='markers', marker=dict(size=4, color='green', opacity=0.5), showlegend=False),
            row=2, col=1
        )
        fig_timeline.add_trace(
            go.Scatter(x=df[anomaly_mask].index, y=df[anomaly_mask]['memory'],
                      mode='markers', marker=dict(size=8, color='red', symbol='x'), showlegend=False),
            row=2, col=1
        )

        # Network
        fig_timeline.add_trace(
            go.Scatter(x=df[normal_mask].index, y=df[normal_mask]['network_io'],
                      mode='markers', marker=dict(size=4, color='orange', opacity=0.5), showlegend=False),
            row=3, col=1
        )
        fig_timeline.add_trace(
            go.Scatter(x=df[anomaly_mask].index, y=df[anomaly_mask]['network_io'],
                      mode='markers', marker=dict(size=8, color='red', symbol='x'), showlegend=False),
            row=3, col=1
        )

        fig_timeline.update_xaxes(title_text="Data Point Index", row=3, col=1)
        fig_timeline.update_yaxes(title_text="CPU (%)", row=1, col=1)
        fig_timeline.update_yaxes(title_text="Memory (%)", row=2, col=1)
        fig_timeline.update_yaxes(title_text="Network I/O (%)", row=3, col=1)

        fig_timeline.update_layout(height=900, showlegend=True)

        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab4:
        st.markdown("#### Parameter Sensitivity Analysis")
        st.markdown("Observe how different epsilon values affect clustering results")

        eps_values = [0.3, 0.5, 0.8, 1.2]

        fig_sensitivity = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'ε = {e}' for e in eps_values],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        feature_columns = ['cpu', 'memory', 'network_io']
        features = df[feature_columns].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        positions = [(1,1), (1,2), (2,1), (2,2)]

        for idx, (eps_val, pos) in enumerate(zip(eps_values, positions)):
            model = DBSCAN(eps=eps_val, min_samples=min_samples)
            temp_labels = model.fit_predict(features_scaled)

            n_clusters = len(set(temp_labels)) - (1 if -1 in temp_labels else 0)
            n_anomalies = list(temp_labels).count(-1)

            # Add scatter
            for label in set(temp_labels):
                mask = temp_labels == label
                color = 'red' if label == -1 else f'rgb({np.random.randint(50,200)},{np.random.randint(50,200)},{np.random.randint(50,200)})'

                fig_sensitivity.add_trace(
                    go.Scatter(
                        x=df[mask]['cpu'],
                        y=df[mask]['memory'],
                        mode='markers',
                        marker=dict(size=5, color=color),
                        name=f'{"Anomaly" if label == -1 else f"C{label}"}',
                        showlegend=False
                    ),
                    row=pos[0], col=pos[1]
                )

            # Update axes
            fig_sensitivity.update_xaxes(title_text="CPU (%)", row=pos[0], col=pos[1])
            fig_sensitivity.update_yaxes(title_text="Memory (%)", row=pos[0], col=pos[1])

            # Add annotation
            fig_sensitivity.add_annotation(
                x=0.5, y=-0.15,
                xref=f'x{idx+1}', yref=f'y{idx+1}',
                text=f'Clusters: {n_clusters} | Anomalies: {n_anomalies}',
                showarrow=False,
                font=dict(size=10)
            )

        fig_sensitivity.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig_sensitivity, use_container_width=True)

    st.divider()

    # === SECTION 4: DATA EXPLORATION ===
    st.markdown('<div class="sub-header">🔍 Data Exploration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Detected Anomalies (Sample)")
        anomalies = df[df['detected_anomaly'] == True][['server_id', 'cpu', 'memory', 'network_io', 'is_anomaly']]
        st.dataframe(anomalies.head(20), use_container_width=True)

    with col2:
        st.markdown("#### Cluster Distribution")
        cluster_dist = df['cluster'].value_counts().sort_index()
        cluster_dist.index = cluster_dist.index.map(lambda x: 'Anomaly' if x == -1 else f'Cluster {x}')

        fig_dist = px.bar(
            x=cluster_dist.index,
            y=cluster_dist.values,
            labels={'x': 'Cluster', 'y': 'Count'},
            title='Distribution of Points Across Clusters',
            color=cluster_dist.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # === SECTION 5: INSIGHTS ===
    st.markdown('<div class="sub-header">💡 Key Insights & Observations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Why DBSCAN?")
        st.markdown("""
        **Advantages for Cloud Anomaly Detection:**
        - ✅ No need to specify number of clusters (unlike K-means)
        - ✅ Automatically identifies outliers as noise points
        - ✅ Can find arbitrarily shaped clusters
        - ✅ Robust to varying cluster densities
        - ✅ Well-suited for infrastructure monitoring

        **Real-World Applications:**
        - Infrastructure health monitoring
        - Auto-scaling trigger detection
        - Performance degradation alerts
        - Resource optimization
        - Incident prediction
        """)

    with col2:
        st.markdown("#### Limitations & Considerations")
        st.markdown("""
        **DBSCAN Limitations:**
        - ⚠️ Sensitive to epsilon (ε) parameter selection
        - ⚠️ Struggles with varying density clusters
        - ⚠️ High-dimensional data requires careful feature engineering
        - ⚠️ Not ideal for real-time streaming (batch-oriented)
        - ⚠️ Performance degrades with very large datasets

        **Production Deployment:**
        - Use HDBSCAN for hierarchical clustering
        - Implement streaming pipeline (Kafka)
        - Integrate with alerting systems
        - Containerize as microservice (Docker/K8s)
        - Add auto-tuning for epsilon
        """)

    st.divider()

    # === SECTION 6: DOWNLOAD ===
    st.markdown('<div class="sub-header">📥 Download Results</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download Clustered Data (CSV)",
            data=csv,
            file_name="clustered_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        anomalies_csv = df[df['detected_anomaly'] == True].to_csv(index=False)
        st.download_button(
            label="⚠️ Download Anomalies Only (CSV)",
            data=anomalies_csv,
            file_name="detected_anomalies.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # Create analysis report
        report = f"""
        DBSCAN Cloud Anomaly Detection Report
        =====================================

        Configuration:
        - Data Points: {len(df)}
        - Servers: {n_servers}
        - Epsilon (ε): {eps:.3f}
        - Min Samples: {min_samples}

        Results:
        - Clusters Found: {analysis['n_clusters']}
        - Anomalies Detected: {analysis['n_anomalies']}
        - Anomaly Rate: {analysis['anomaly_percentage']:.2f}%

        Performance:
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}

        Confusion Matrix:
        - True Positives: {tp}
        - True Negatives: {tn}
        - False Positives: {fp}
        - False Negatives: {fn}
        """

        st.download_button(
            label="📊 Download Analysis Report (TXT)",
            data=report,
            file_name="analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    # Initial state - show instructions
    st.info("👈 Configure parameters in the sidebar and click **'Generate & Analyze'** to start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Data Generation")
        st.markdown("""
        - Synthetic cloud metrics
        - Configurable servers
        - Realistic anomaly injection
        - Multiple resource types
        """)

    with col2:
        st.markdown("### 🤖 DBSCAN Clustering")
        st.markdown("""
        - Automatic parameter tuning
        - Density-based detection
        - Noise point identification
        - Performance evaluation
        """)

    with col3:
        st.markdown("### 📈 Visualizations")
        st.markdown("""
        - Interactive 2D/3D plots
        - Timeline analysis
        - Parameter sensitivity
        - Confusion matrix
        """)

    st.divider()

    st.markdown("### 🎓 Project Overview")
    st.markdown("""
    This application demonstrates **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
    for detecting anomalies in cloud infrastructure monitoring data.

    **Key Features:**
    - Real-time parameter adjustment
    - Comprehensive performance metrics
    - Interactive visualizations
    - Downloadable results

    **Use Cases:**
    - Infrastructure health monitoring
    - Performance anomaly detection
    - Auto-scaling optimization
    - Incident prediction
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem 0;'>
    <strong>Cloud Anomaly Detection with DBSCAN</strong> | M.Tech AI/ML Project<br>
    Density-Based Clustering Application for Infrastructure Monitoring
</div>
""", unsafe_allow_html=True)
