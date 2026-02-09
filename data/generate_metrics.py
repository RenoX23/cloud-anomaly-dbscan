import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_cloud_metrics(n_points=2000, n_servers=5, anomaly_rate=0.05):
    """
    Generate synthetic cloud resource metrics with injected anomalies

    Parameters:
    -----------
    n_points: int
        Number of time points to generate
    n_servers: int
        Number of virtual servers to simulate
    anomaly_rate: float
        Percentage of data points that should be anomalies

    Returns:
    --------
    pd.DataFrame with columns: timestamp, server_id, cpu, memory, network_io, is_anomaly
    """

    np.random.seed(42)

    # Generate timestamps (1 data point per minute)
    start_time = datetime.now() - timedelta(hours=n_points//60)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]

    data = []

    for i, ts in enumerate(timestamps):
        for server_id in range(1, n_servers + 1):
            # Normal behavior - different servers have different baselines
            base_cpu = 40 + (server_id * 5)  # Servers have different typical loads
            base_memory = 50 + (server_id * 3)

            cpu = np.random.normal(base_cpu, 10)
            memory = np.random.normal(base_memory, 12)
            network_io = np.random.normal(30, 8)

            # Clip to valid ranges
            cpu = np.clip(cpu, 0, 100)
            memory = np.clip(memory, 0, 100)
            network_io = np.clip(network_io, 0, 100)

            is_anomaly = False

            # Inject different types of anomalies
            if np.random.random() < anomaly_rate:
                anomaly_type = np.random.choice([
                    'cpu_spike',
                    'memory_spike',
                    'resource_exhaustion',
                    'network_spike'
                ])

                if anomaly_type == 'cpu_spike':
                    cpu = np.random.uniform(85, 100)
                elif anomaly_type == 'memory_spike':
                    memory = np.random.uniform(85, 100)
                elif anomaly_type == 'resource_exhaustion':
                    cpu = np.random.uniform(85, 100)
                    memory = np.random.uniform(85, 100)
                elif anomaly_type == 'network_spike':
                    network_io = np.random.uniform(80, 100)

                is_anomaly = True

            data.append({
                'timestamp': ts,
                'server_id': f'server_{server_id}',
                'cpu': round(cpu, 2),
                'memory': round(memory, 2),
                'network_io': round(network_io, 2),
                'is_anomaly': is_anomaly
            })

    df = pd.DataFrame(data)
    return df

def save_metrics(df, filepath='sample_metrics.csv'):
    """Save generated metrics to CSV"""
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df)} data points to {filepath}")

if __name__ == "__main__":
    print("="*60)
    print("Cloud Metrics Data Generator")
    print("="*60)

    df = generate_cloud_metrics(n_points=2000, n_servers=5, anomaly_rate=0.05)

    print(f"\nDataset Statistics:")
    print(f"  Total data points: {len(df)}")
    print(f"  Number of servers: {df['server_id'].nunique()}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Injected anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.2f}%)")

    print(f"\nFeature Statistics:")
    print(df[['cpu', 'memory', 'network_io']].describe())

    print(f"\nFirst 10 rows:")
    print(df.head(10))

    save_metrics(df, 'data/sample_metrics.csv')
