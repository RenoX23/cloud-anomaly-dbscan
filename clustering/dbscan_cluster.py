import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

class CloudAnomalyDetector:
    """
    DBSCAN-based anomaly detector for cloud resource metrics
    """

    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize detector

        Parameters:
        -----------
        eps: float
            Maximum distance between two samples for one to be considered
            as in the neighborhood of the other
        min_samples: int
            Number of samples in a neighborhood for a point to be considered
            as a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = ['cpu', 'memory', 'network_io']

    def fit_predict(self, df):
        """
        Apply DBSCAN clustering to detect anomalies

        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with cloud metrics

        Returns:
        --------
        np.array: Cluster labels (-1 indicates anomaly)
        """
        # Extract features
        features = df[self.feature_columns].values

        # Normalize features (critical for DBSCAN)
        features_scaled = self.scaler.fit_transform(features)

        # Apply DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(features_scaled)

        return labels

    def analyze_clusters(self, df, labels):
        """
        Analyze clustering results

        Returns:
        --------
        dict: Analysis statistics
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_anomalies = list(labels).count(-1)
        n_noise = n_anomalies

        analysis = {
            'n_clusters': n_clusters,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': (n_anomalies / len(labels)) * 100,
            'cluster_sizes': {}
        }

        # Get cluster sizes
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:
                size = list(labels).count(label)
                analysis['cluster_sizes'][f'cluster_{label}'] = size

        return analysis

    def get_anomaly_details(self, df, labels):
        """
        Get detailed information about detected anomalies
        """
        df_copy = df.copy()
        df_copy['cluster'] = labels

        anomalies = df_copy[df_copy['cluster'] == -1]

        return anomalies

def find_optimal_epsilon(df, min_samples=5, feature_columns=['cpu', 'memory', 'network_io']):
    """
    Find optimal epsilon using k-distance graph method

    Returns suggested epsilon value
    """
    from sklearn.neighbors import NearestNeighbors

    features = df[feature_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Compute k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features_scaled)
    distances, indices = neighbors_fit.kneighbors(features_scaled)

    # Sort distances
    distances = np.sort(distances[:, -1], axis=0)

    # Find elbow point (simple method - take 95th percentile)
    suggested_eps = np.percentile(distances, 95)

    return suggested_eps, distances

if __name__ == "__main__":
    print("="*60)
    print("DBSCAN Cloud Anomaly Detection")
    print("="*60)

    # Load data
    df = pd.read_csv('../data/sample_metrics.csv')
    print(f"\n✓ Loaded {len(df)} data points")

    # Find optimal epsilon
    print("\n[1] Finding optimal epsilon...")
    suggested_eps, distances = find_optimal_epsilon(df)
    print(f"    Suggested epsilon: {suggested_eps:.3f}")

    # Apply DBSCAN with suggested epsilon
    print(f"\n[2] Applying DBSCAN (eps={suggested_eps:.3f}, min_samples=5)...")
    detector = CloudAnomalyDetector(eps=suggested_eps, min_samples=5)
    labels = detector.fit_predict(df)

    # Analyze results
    print("\n[3] Analyzing results...")
    analysis = detector.analyze_clusters(df, labels)

    print(f"\n    Clusters found: {analysis['n_clusters']}")
    print(f"    Anomalies detected: {analysis['n_anomalies']} ({analysis['anomaly_percentage']:.2f}%)")
    print(f"    Cluster sizes: {analysis['cluster_sizes']}")

    # Get anomaly details
    anomalies = detector.get_anomaly_details(df, labels)
    print(f"\n[4] Sample anomalies detected:")
    print(anomalies[['timestamp', 'server_id', 'cpu', 'memory', 'network_io']].head(10))

    # Save results
    df['cluster'] = labels
    df['detected_anomaly'] = (labels == -1)
    df.to_csv('../data/clustered_metrics.csv', index=False)
    print(f"\n✓ Results saved to '../data/clustered_metrics.csv'")
