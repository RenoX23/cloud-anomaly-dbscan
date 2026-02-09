"""
Cloud Resource Anomaly Detection using DBSCAN
Complete Demo Implementation
"""

import pandas as pd
import numpy as np
import sys
import os

# Add paths
sys.path.append('data')
sys.path.append('clustering')
sys.path.append('visualization')

from generate_metrics import generate_cloud_metrics, save_metrics
from dbscan_cluster import CloudAnomalyDetector, find_optimal_epsilon
from plot_clusters import (plot_2d_clusters, plot_3d_clusters,
                           plot_parameter_sensitivity, plot_anomaly_timeline,
                           plot_confusion_matrix)

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def main():
    print_header("CLOUD ANOMALY DETECTION WITH DBSCAN")
    print("Density-Based Clustering for Infrastructure Monitoring")
    print("M.Tech AI/ML Project Demonstration\n")

    # ========== STEP 1: DATA GENERATION ==========
    print_header("STEP 1: GENERATING SYNTHETIC CLOUD METRICS")

    df = generate_cloud_metrics(n_points=2000, n_servers=5, anomaly_rate=0.05)

    print(f"\n✓ Generated Dataset:")
    print(f"  • Total data points: {len(df)}")
    print(f"  • Number of servers: {df['server_id'].nunique()}")
    print(f"  • Features: CPU, Memory, Network I/O")
    print(f"  • Injected anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.2f}%)")

    print(f"\n  Feature Statistics:")
    print(df[['cpu', 'memory', 'network_io']].describe().round(2))

    # Save data
    os.makedirs('data', exist_ok=True)
    save_metrics(df, 'data/sample_metrics.csv')

    # ========== STEP 2: OPTIMAL PARAMETER SELECTION ==========
    print_header("STEP 2: FINDING OPTIMAL DBSCAN PARAMETERS")

    suggested_eps, distances = find_optimal_epsilon(df, min_samples=5)
    print(f"\n✓ Parameter Analysis:")
    print(f"  • Suggested epsilon (ε): {suggested_eps:.3f}")
    print(f"  • Min samples: 5")
    print(f"  • Rationale: Based on k-distance graph elbow method")

    # ========== STEP 3: APPLYING DBSCAN ==========
    print_header("STEP 3: APPLYING DBSCAN CLUSTERING")

    detector = CloudAnomalyDetector(eps=suggested_eps, min_samples=5)
    labels = detector.fit_predict(df)

    analysis = detector.analyze_clusters(df, labels)

    print(f"\n✓ Clustering Results:")
    print(f"  • Clusters identified: {analysis['n_clusters']}")
    print(f"  • Anomalies detected: {analysis['n_anomalies']} ({analysis['anomaly_percentage']:.2f}%)")
    print(f"  • Cluster distribution:")
    for cluster, size in analysis['cluster_sizes'].items():
        print(f"    - {cluster}: {size} points")

    # ========== STEP 4: ANOMALY DETAILS ==========
    print_header("STEP 4: ANALYZING DETECTED ANOMALIES")

    anomalies = detector.get_anomaly_details(df, labels)

    print(f"\n✓ Sample Detected Anomalies (First 10):")
    print(anomalies[['server_id', 'cpu', 'memory', 'network_io']].head(10).to_string(index=False))

    print(f"\n  Anomaly Characteristics:")
    print(f"  • Average CPU in anomalies: {anomalies['cpu'].mean():.2f}%")
    print(f"  • Average Memory in anomalies: {anomalies['memory'].mean():.2f}%")
    print(f"  • Average Network I/O in anomalies: {anomalies['network_io'].mean():.2f}%")

    # Save clustered results
    df['cluster'] = labels
    df['detected_anomaly'] = (labels == -1)
    df.to_csv('data/clustered_metrics.csv', index=False)
    print(f"\n✓ Results saved to 'data/clustered_metrics.csv'")

    # ========== STEP 5: MODEL EVALUATION ==========
    print_header("STEP 5: MODEL EVALUATION")

    from sklearn.metrics import confusion_matrix, classification_report

    actual = df['is_anomaly'].astype(int)
    detected = (labels == -1).astype(int)

    cm = confusion_matrix(actual, detected)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n✓ Performance Metrics:")
    print(f"  • Accuracy: {accuracy:.3f}")
    print(f"  • Precision: {precision:.3f} (of detected anomalies, how many were real)")
    print(f"  • Recall: {recall:.3f} (of real anomalies, how many were detected)")
    print(f"  • F1-Score: {f1:.3f}")

    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {tn:4d}  |  False Positives: {fp:4d}")
    print(f"    False Negatives: {fn:4d}  |  True Positives:  {tp:4d}")

    # ========== STEP 6: VISUALIZATION ==========
    print_header("STEP 6: GENERATING VISUALIZATIONS")

    os.makedirs('visualization', exist_ok=True)

    print("\n✓ Creating visualizations...")

    # 2D Plot
    print("  [1/5] 2D cluster scatter plot...")
    fig1 = plot_2d_clusters(df, labels)
    fig1.savefig('visualization/2d_clusters.png', dpi=300, bbox_inches='tight')

    # 3D Plot
    print("  [2/5] 3D cluster visualization...")
    fig2 = plot_3d_clusters(df, labels)
    fig2.savefig('visualization/3d_clusters.png', dpi=300, bbox_inches='tight')

    # Parameter sensitivity
    print("  [3/5] Parameter sensitivity analysis...")
    fig3 = plot_parameter_sensitivity(df)
    fig3.savefig('visualization/parameter_sensitivity.png', dpi=300, bbox_inches='tight')

    # Timeline
    print("  [4/5] Anomaly timeline...")
    fig4 = plot_anomaly_timeline(df, labels)
    fig4.savefig('visualization/anomaly_timeline.png', dpi=300, bbox_inches='tight')

    # Confusion matrix
    print("  [5/5] Confusion matrix...")
    fig5 = plot_confusion_matrix(df, labels)
    if fig5:
        fig5.savefig('visualization/confusion_matrix.png', dpi=300, bbox_inches='tight')

    print("\n✓ All visualizations saved to 'visualization/' directory")

    # ========== STEP 7: KEY INSIGHTS ==========
    print_header("STEP 7: KEY INSIGHTS & CONCLUSIONS")

    print("\n✓ Why DBSCAN for Cloud Anomaly Detection?")
    print("  1. No need to specify number of clusters (unlike K-means)")
    print("  2. Automatically identifies outliers/anomalies as noise points")
    print("  3. Can find arbitrarily shaped clusters")
    print("  4. Robust to varying cluster densities")

    print("\n✓ Real-World Applications:")
    print("  • Infrastructure health monitoring")
    print("  • Auto-scaling trigger detection")
    print("  • Performance degradation alerts")
    print("  • Resource optimization")

    print("\n✓ Limitations & Considerations:")
    print("  • Sensitive to epsilon (ε) parameter selection")
    print("  • Struggles with varying density clusters")
    print("  • High-dimensional data requires careful feature engineering")
    print("  • Not ideal for real-time streaming (batch-oriented)")

    print("\n✓ Future Enhancements:")
    print("  • Implement HDBSCAN for hierarchical density-based clustering")
    print("  • Add real-time streaming data pipeline")
    print("  • Integrate with alerting system (email/Slack)")
    print("  • Deploy as containerized microservice")

    print_header("DEMO COMPLETE")
    print("\nAll outputs generated:")
    print("  • Data: data/sample_metrics.csv, data/clustered_metrics.csv")
    print("  • Visualizations: visualization/*.png")
    print("\nYou can now:")
    print("  1. Review the generated visualizations")
    print("  2. Analyze the clustered_metrics.csv for detailed results")
    print("  3. Prepare presentation slides using these outputs")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
