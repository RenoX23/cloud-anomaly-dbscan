import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def plot_2d_clusters(df, labels, feature_x='cpu', feature_y='memory', title='DBSCAN Clustering Results'):
    """
    Plot 2D scatter of clusters
    """
    plt.figure(figsize=(12, 8))

    # Get unique labels
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Anomalies in red with X marker
            color = 'red'
            marker = 'X'
            label_name = 'Anomaly'
            size = 100
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
            size = 50

        mask = labels == label
        plt.scatter(
            df[mask][feature_x],
            df[mask][feature_y],
            c=[color],
            label=label_name,
            marker=marker,
            s=size,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'{feature_x.upper()} (%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'{feature_y.upper()} (%)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()

def plot_3d_clusters(df, labels):
    """
    Plot 3D scatter of clusters (CPU, Memory, Network)
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'red'
            marker = 'X'
            label_name = 'Anomaly'
            size = 100
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
            size = 30

        mask = labels == label
        ax.scatter(
            df[mask]['cpu'],
            df[mask]['memory'],
            df[mask]['network_io'],
            c=[color],
            label=label_name,
            marker=marker,
            s=size,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('CPU (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory (%)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Network I/O (%)', fontsize=11, fontweight='bold')
    ax.set_title('3D DBSCAN Clustering - Cloud Resources', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)

    return fig

def plot_parameter_sensitivity(df, eps_values=[0.3, 0.5, 0.8, 1.2], min_samples=5):
    """
    Show how different epsilon values affect clustering
    """
    feature_columns = ['cpu', 'memory', 'network_io']
    features = df[feature_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, eps in enumerate(eps_values):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(features_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_anomalies = list(labels).count(-1)

        ax = axes[idx]

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'red'
                marker = 'X'
                size = 80
            else:
                marker = 'o'
                size = 40

            mask = labels == label
            ax.scatter(
                df[mask]['cpu'],
                df[mask]['memory'],
                c=[color],
                marker=marker,
                s=size,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )

        ax.set_xlabel('CPU (%)', fontsize=10)
        ax.set_ylabel('Memory (%)', fontsize=10)
        ax.set_title(f'eps={eps} | Clusters: {n_clusters} | Anomalies: {n_anomalies}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('DBSCAN Parameter Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig

def plot_anomaly_timeline(df, labels):
    """
    Show anomalies over time
    """
    df_copy = df.copy()
    df_copy['cluster'] = labels
    df_copy['is_detected_anomaly'] = (labels == -1)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # CPU timeline
    axes[0].plot(df_copy.index, df_copy['cpu'], alpha=0.5, color='blue', label='Normal')
    anomaly_indices = df_copy[df_copy['is_detected_anomaly']].index
    axes[0].scatter(anomaly_indices, df_copy.loc[anomaly_indices, 'cpu'],
                    color='red', s=50, marker='X', label='Anomaly', zorder=5)
    axes[0].set_ylabel('CPU (%)', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Resource Metrics with Detected Anomalies', fontsize=14, fontweight='bold')

    # Memory timeline
    axes[1].plot(df_copy.index, df_copy['memory'], alpha=0.5, color='green', label='Normal')
    axes[1].scatter(anomaly_indices, df_copy.loc[anomaly_indices, 'memory'],
                    color='red', s=50, marker='X', label='Anomaly', zorder=5)
    axes[1].set_ylabel('Memory (%)', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Network timeline
    axes[2].plot(df_copy.index, df_copy['network_io'], alpha=0.5, color='orange', label='Normal')
    axes[2].scatter(anomaly_indices, df_copy.loc[anomaly_indices, 'network_io'],
                    color='red', s=50, marker='X', label='Anomaly', zorder=5)
    axes[2].set_ylabel('Network I/O (%)', fontweight='bold')
    axes[2].set_xlabel('Data Point Index', fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig

def plot_confusion_matrix(df, labels):
    """
    Compare actual vs detected anomalies (if ground truth available)
    """
    if 'is_anomaly' not in df.columns:
        print("Ground truth not available, skipping confusion matrix")
        return None

    from sklearn.metrics import confusion_matrix, classification_report

    actual = df['is_anomaly'].astype(int)
    detected = (labels == -1).astype(int)

    cm = confusion_matrix(actual, detected)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Detected', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Actual vs Detected Anomalies', fontsize=14, fontweight='bold')

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    ax.text(1.5, 0.5, metrics_text, fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return fig

if __name__ == "__main__":
    print("="*60)
    print("Generating Visualizations")
    print("="*60)

    # Load clustered data
    df = pd.read_csv('../data/clustered_metrics.csv')
    labels = df['cluster'].values

    print(f"\n✓ Loaded {len(df)} data points with cluster labels")

    # Generate all plots
    print("\n[1] Creating 2D cluster plot...")
    fig1 = plot_2d_clusters(df, labels)
    fig1.savefig('../visualization/2d_clusters.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved to visualization/2d_clusters.png")

    print("\n[2] Creating 3D cluster plot...")
    fig2 = plot_3d_clusters(df, labels)
    fig2.savefig('../visualization/3d_clusters.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved to visualization/3d_clusters.png")

    print("\n[3] Creating parameter sensitivity analysis...")
    fig3 = plot_parameter_sensitivity(df)
    fig3.savefig('../visualization/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved to visualization/parameter_sensitivity.png")

    print("\n[4] Creating anomaly timeline...")
    fig4 = plot_anomaly_timeline(df, labels)
    fig4.savefig('../visualization/anomaly_timeline.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved to visualization/anomaly_timeline.png")

    print("\n[5] Creating confusion matrix...")
    fig5 = plot_confusion_matrix(df, labels)
    if fig5:
        fig5.savefig('../visualization/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved to visualization/confusion_matrix.png")

    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
