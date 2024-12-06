"""
This module compares various clustering algorithms (DBSCAN, OPTICS, HDBSCAN, KMeans, Agglomerative, GMM)
on log data. It tokenizes and vectorizes the logs, optimizes parameters like eps for DBSCAN,
estimates the number of clusters for some algorithms, and evaluates clustering quality using
Silhouette Score and other metrics. Finally, it ranks the methods and visualizes the results.

Algorithms included:
- DBSCAN (density-based, requires eps optimization)
- OPTICS (density-based, parameter max_eps chosen by DBSCAN's best eps)
- HDBSCAN (density-based, hierarchical, no preset number of clusters)
- KMeans (centroid-based, requires a chosen number of clusters)
- Agglomerative Clustering (hierarchical, requires a chosen number of clusters)
- GMM (Gaussian Mixture Model, probabilistic, requires a chosen number of clusters)

The results are saved in a .txt file and a PNG plot for metrics. Visualization of clusters is also provided.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import time
import warnings
from collections import OrderedDict
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import DBSCAN, OPTICS, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import hdbscan
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure the compare_plots directory exists
if not os.path.exists('compare_plots'):
    os.makedirs('compare_plots')

def tokenize(log_content, tokenize_pattern=r'[ ,|]', removeDight=True):
    """
    Tokenize and preprocess a single log message.

    This function:
    1. Replaces various patterns (dates, IPs, URLs, log levels, hex, numbers) in the log
       with placeholder tokens (e.g., <DATE_TIME>, <IP>, <URL>, <LEVEL>, <HEX>, <NUM>).
    2. Splits the log based on a given regex pattern.
    3. Optionally removes tokens containing digits if removeDight=True.
    4. Preserves token order by using positional indexing.
    
    Parameters
    ----------
    log_content : str
        The raw log message.
    tokenize_pattern : str, optional
        Regex pattern to split the log into tokens, by default r'[ ,|]'.
    removeDight : bool, optional
        If True, remove tokens containing digits. By default True.

    Returns
    -------
    list of str
        A list of tokens representing the processed log message.
    """
    # Enhanced patterns
    log_content = re.sub(
        r'\d{2,4}[-/:]\d{1,2}[-/:]\d{1,4}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?',
        '<DATE_TIME>', log_content)
    log_content = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', log_content)
    log_content = re.sub(
        r'http[s]?://\S+', '<URL>', log_content)
    log_content = re.sub(
        r'\b(ERROR|WARN|INFO|DEBUG)\b', '<LEVEL>', log_content)
    log_content = re.sub(
        r'0x[0-9a-fA-F]+', '<HEX>', log_content)
    log_content = re.sub(
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', '<NUM>', log_content)

    words = re.split(tokenize_pattern, log_content)
    new_words = []
    position = 0
    
    for word in words:
        if not word:
            continue
        if '=' in word:
            ws = word.split('=')
            if len(ws) <= 2:
                new_words.append((ws[0], position))
        elif not (removeDight and re.search(r'\d', word)) and '/' not in word.lower():
            new_words.append((word, position))
        position += len(word) + 1
    
    if not new_words:
        return [re.sub(r'\d+(\.\d+)?', '0', log_content)]
    
    # Return tokens with preserved order
    return [token for token, _ in new_words]


def vectorize(tokenized_logs):
    """
    Vectorize tokenized logs using TF-IDF and normalize the resulting vectors.

    Parameters
    ----------
    tokenized_logs : list of list of str
        Each element is the tokenized form of a log message.

    Returns
    -------
    scipy.sparse matrix
        A normalized TF-IDF sparse matrix representing the log corpus.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        lowercase=False
    )
    tfidf_matrix = vectorizer.fit_transform(tokenized_logs)
    tfidf_matrix = normalize(tfidf_matrix)
    if not tfidf_matrix.nnz:
        print("Warning: TF-IDF matrix contains all zeros.")
    return tfidf_matrix


def optimize_eps_for_dbscan(vectorized_logs, min_samples=5):
    """
    Optimize eps for DBSCAN by selecting the value that yields the highest Silhouette Score.
    """
    if sparse.issparse(vectorized_logs):
        X_dense = vectorized_logs.toarray()
    else:
        X_dense = vectorized_logs

    eps_candidates = np.linspace(0.05, 0.5, 10)
    best_eps = None
    best_score = -1.0

    for eps_candidate in eps_candidates:
        clusterer = DBSCAN(eps=eps_candidate, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        labels = clusterer.fit_predict(vectorized_logs)
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            # Check if multiple real clusters
            real_clusters = [lbl for lbl in unique_labels if lbl != -1]
            if len(real_clusters) > 1:
                score = silhouette_score(X_dense, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps_candidate

    # Fallback if no suitable eps found
    if best_eps is None:
        best_eps = 0.1
        print(f"No valid eps found for DBSCAN with multiple clusters. Using default eps={best_eps}")
    else:
        print(f"Selected eps={best_eps} for DBSCAN based on Silhouette Score={best_score:.4f}")

    return best_eps

def estimate_n_clusters(vectorized_logs, max_clusters=50):
    """
    Estimate optimal number of clusters using the elbow method
    and dataset characteristics.
    """
    n_samples = vectorized_logs.shape[0]
    
    # For very small datasets
    if n_samples < 100:
        return min(int(np.sqrt(n_samples)), max_clusters)
        
    # For larger datasets, use elbow method with sample
    sample_size = min(1000, n_samples)  # Cap sample size for efficiency
    if sparse.issparse(vectorized_logs):
        sample_data = vectorized_logs[:sample_size].toarray()
    else:
        sample_data = vectorized_logs[:sample_size]

    distortions = []
    K = range(2, min(20, max_clusters))
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=2)
        kmeans.fit(sample_data)
        distortions.append(kmeans.inertia_)
    
    # Find elbow point using differences
    diffs = np.diff(distortions)
    elbow_idx = np.argmax(diffs) + 2  # Add 2 because we started from K=2
    
    # Apply constraints
    optimal_k = min(
        max(elbow_idx, 2),  # At least 2 clusters
        int(np.sqrt(n_samples/2)),  # Rule of thumb upper bound
        max_clusters  # Hard upper limit
    )
    
    return optimal_k

def compare_clustering_methods(vectorized_logs, dataset_name='Dataset', min_samples=5):
    """
    Compare multiple clustering algorithms on the given dataset and rank them by Silhouette Score.

    Algorithms: DBSCAN, OPTICS, HDBSCAN, KMeans, Agglomerative, GMM.
    - DBSCAN eps optimized by Silhouette Score.
    - OPTICS uses DBSCAN's best eps for max_eps.
    - HDBSCAN does not need a cluster count.
    - KMeans, Agglomerative, GMM need estimated cluster count.
    - Results are saved and a visualization is produced.

    Parameters
    ----------
    vectorized_logs : scipy.sparse matrix or np.ndarray
        Vectorized log data.
    dataset_name : str
        Name of the dataset for output file naming.
    min_samples : int, optional
        Parameter for DBSCAN/HDBSCAN/OPTICS, by default 5.

    Returns
    -------
    pd.DataFrame
        DataFrame with each method's clustering metrics and results.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import time

    # Convert to dense if sparse
    if sparse.issparse(vectorized_logs):
        vectorized_logs_dense = vectorized_logs.toarray()
    else:
        vectorized_logs_dense = vectorized_logs

    # Optimize eps for DBSCAN
    best_eps_dbscan = optimize_eps_for_dbscan(vectorized_logs, min_samples=min_samples)

    # For OPTICS, we will just pick a default max_eps or rely on the default
    # This avoids the time-consuming eps search for OPTICS
    default_max_eps_optics = 0.5
    n_clusters = estimate_n_clusters(vectorized_logs)
    # Define clustering methods with the chosen eps parameters
    clustering_methods = {
        'DBSCAN': DBSCAN(eps=best_eps_dbscan, min_samples=min_samples, metric='euclidean', n_jobs=-1),
        'OPTICS': OPTICS(min_samples=min_samples, metric='euclidean', max_eps=best_eps_dbscan, n_jobs=-1),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='euclidean'),
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'GMM': GaussianMixture(n_components=n_clusters, random_state=42)
    }

    results = []
    all_labels = {}

    print("\nComparing clustering methods:")
    print("-" * 80)

    for name, clusterer in clustering_methods.items():
        try:
            start_time = time.time()
            if name in ['DBSCAN', 'OPTICS']:
                clusterer.fit(vectorized_logs)
            else:
                clusterer.fit(vectorized_logs_dense)
            clustering_time = time.time() - start_time

            labels = clusterer.labels_ if hasattr(clusterer, 'labels_') else clusterer.predict(vectorized_logs_dense)

            all_labels[name] = labels

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1) if -1 in labels else 0

            metrics = {}
            if n_clusters > 1:
                metrics['silhouette'] = silhouette_score(vectorized_logs_dense, labels)
                metrics['calinski'] = calinski_harabasz_score(vectorized_logs_dense, labels)
                metrics['davies'] = davies_bouldin_score(vectorized_logs_dense, labels)
            else:
                metrics['silhouette'] = -1
                metrics['calinski'] = -1
                metrics['davies'] = -1

            results.append({
                'Method': name,
                'Clusters': n_clusters,
                'Noise Points': n_noise,
                'Time (s)': clustering_time,
                'Silhouette Score': metrics.get('silhouette'),
                'Calinski-Harabasz Score': metrics.get('calinski'),
                'Davies-Bouldin Score': metrics.get('davies')
            })

            print(f"{name}: Clusters={n_clusters}, Noise Points={n_noise}, Time={clustering_time:.2f}s")

        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue

    # Create DataFrame and display results
    df_results = pd.DataFrame(results)
    print("\nClustering Results:")
    print(df_results.to_string(index=False))

    # Save metrics to a .txt file
    metrics_file = os.path.join('compare_plots', f'{dataset_name}_clustering_metrics.txt')
    df_results.to_csv(metrics_file, sep='\t', index=False)
    print(f"\nClustering metrics saved to {metrics_file}")

    # Visualize results and save plots
    plot_clustering_metrics(df_results, dataset_name)
    visualize_clusters(vectorized_logs_dense, all_labels, dataset_name)

    return df_results


def plot_clustering_metrics(df_results, dataset_name):
    """
    Plot and save clustering metrics.
    """
    metrics = ['Clusters', 'Noise Points', 'Time (s)', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
    num_metrics = len(metrics)

    plt.figure(figsize=(15, 5 * ((num_metrics + 1) // 2)))

    for i, metric in enumerate(metrics, 1):
        plt.subplot((num_metrics + 1) // 2, 2, i)
        plt.bar(df_results['Method'], df_results[metric])
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()

    plt.suptitle(f'Clustering Metrics Comparison - {dataset_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_file = os.path.join('compare_plots', f'{dataset_name}_clustering_metrics.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Clustering metrics plot saved to {plot_file}")


def visualize_clusters(vectorized_logs_dense, all_labels, dataset_name):
    """
    Visualize clusters using PCA for dimensionality reduction.
    """
    num_samples, num_features = vectorized_logs_dense.shape
    if num_samples < 2:
        print("Not enough samples for PCA visualization. Skipping.")
        return
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectorized_logs_dense)

    num_methods = len(all_labels)
    plt.figure(figsize=(5 * num_methods, 5))

    for i, (name, labels) in enumerate(all_labels.items(), 1):
        plt.subplot(1, num_methods, i)
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20', s=10)
        plt.title(f'{name}\n({len(set(labels)) - (1 if -1 in labels else 0)} clusters)')
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f'Cluster Visualization - {dataset_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_file = os.path.join('compare_plots', f'{dataset_name}_cluster_visualization.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Cluster visualization plot saved to {plot_file}")


def process_all_datasets(datasets_dir='../datasets/loghub-2k'):
    """
    Process all .log files in the datasets directory.
    """
    # List to store recommendations
    recommendations = []

    # Loop through all directories in datasets_dir
    datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]

    for dataset in tqdm(datasets, desc='Processing Datasets'):
        dataset_path = os.path.join(datasets_dir, dataset)
        # Find .log files in the dataset directory
        log_files = [f for f in os.listdir(dataset_path) if f.endswith('.log')]

        for log_file in log_files:
            file_path = os.path.join(dataset_path, log_file)
            dataset_name = os.path.splitext(log_file)[0]

            print(f"\nProcessing dataset: {dataset_name}")
            # Read log data
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                log_data = file.readlines()

            # Tokenize logs
            tokenized_logs = [tokenize(log) for log in log_data]

            # Vectorize logs
            tfidf_matrix = vectorize(tokenized_logs)

            # Compare clustering methods
            comparison_results = compare_clustering_methods(tfidf_matrix, dataset_name=dataset_name)

            # Rank all methods by Silhouette Score
            ranked_results = comparison_results.sort_values('Silhouette Score', ascending=False)
            ranked_results['Rank'] = range(1, len(ranked_results) + 1)

            # Store all methods with ranks
            recommendations.append({
                'Dataset': dataset_name,
                'Ranked Methods': ranked_results[['Rank', 'Method', 'Silhouette Score']].values.tolist()
            })

    # Write recommendations to file, showing all methods with their rank
    with open('cluster_recommendation.txt', 'w') as f:
        f.write("Ranked Methods per Dataset:\n\n")
        for rec in recommendations:
            f.write(f"Dataset: {rec['Dataset']}\n")
            for rank, method, score in rec['Ranked Methods']:
                f.write(f"  Rank: {rank}, Method: {method}, Silhouette Score: {score:.4f}\n")
            f.write("\n")

    print("\nCluster recommendations saved to 'cluster_recommendation.txt'")


if __name__ == "__main__":
    # Ensure the required packages are installed
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scikit-learn', 'hdbscan', 'kneed', 'tqdm'
    ]
    import subprocess
    import sys

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Process all datasets
    process_all_datasets()
