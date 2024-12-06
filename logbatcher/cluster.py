from collections import OrderedDict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, OPTICS
from logbatcher.sample import group_samples_clustering, dpp_sample
import random
import hdbscan
from sklearn.preprocessing import normalize
from scipy import sparse
from sklearn.metrics import silhouette_score
import numpy as np
class Cluster:
    def __init__(self):
        self.logs = []
        self.batch_logs = []
        self.indexs = []
        self.size = 0
        

    def append_log(self, log, index):
        self.logs.append(log)
        self.indexs.append(index)
        self.size += 1
    
    def batching(self, batch_size=10, sample_method="dpp"):
        self.batch_logs = list(OrderedDict.fromkeys(self.logs)) # remove duplicates
        if len(self.batch_logs) > batch_size:
            self.sample(batch_size, sample_method)

    def sample(self, batch_size, sample_method):
        # vetorize logs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.batch_logs)
        tfidf_matrix = tfidf_matrix.toarray()

        # sample
        if sample_method == "dpp":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            result = dpp_sample(similarity_matrix, batch_size)
        elif sample_method == "random":
            random.seed(0)
            result = random.sample(range(0, len(self.batch_logs)), batch_size)
        elif sample_method == "similar":
            result = group_samples_clustering(tfidf_matrix, batch_size)[0]
        else:
            raise ValueError("Invalid sample method")
        self.batch_logs = [self.batch_logs[i] for i in result]
        return

def tokenize(log_content, tokenize_pattern=r'[ ,|]', removeDight=True):
    """
    Tokenize log messages, replacing various patterns with placeholders.
    
    The function:
    - Replaces dates, IPs, URLs, log levels, hex values, and numbers with special tokens.
    - Splits the log message by delimiters (spaces, commas, pipes).
    - Attempts to preserve token position ordering.
    - Filters out certain tokens (like digits if removeDight=True).
    
    Parameters
    ----------
    log_content : str
        The raw log message.
    tokenize_pattern : str, optional
        Regex pattern to split the log, by default r'[ ,|]'.
    removeDight : bool, optional
        If True, remove tokens containing digits (except replaced placeholders), by default True.

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
    Vectorize tokenized logs using TF-IDF and then normalize the resulting vectors.
    
    Parameters
    ----------
    tokenized_logs : list of list of str
        Each element is a tokenized log message.

    Returns
    -------
    sparse matrix
        A normalized TF-IDF matrix representing the log corpus.
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
        best_eps = 0.5
        print(f"No valid eps found for DBSCAN with multiple clusters. Using default eps={best_eps}")
    else:
        print(f"Selected eps={best_eps} for DBSCAN based on Silhouette Score={best_score:.4f}")

    return best_eps

def dbscan_clustering(vectorized_logs, eps, min_samples=5):
    data = vectorized_logs
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    clusterer.fit(data)
    labels = clusterer.labels_
    return labels


def hdbscan_clustering(vectorized_logs, min_cluster_size=5):
    data = vectorized_logs.toarray()  # HDBSCAN requires dense input
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(data)
    return labels


def optics_clustering(vectorized_logs, min_samples=5):
    data = vectorized_logs.toarray()
    # Check for zero variance
    if np.all(data == data[0]):
        print("All data points are identical. Skipping OPTICS clustering.")
        return np.full(len(data), -1)
    clusterer = OPTICS(min_samples=min_samples, metric='euclidean')
    clusterer.fit(data)
    labels = clusterer.labels_
    return labels

def consensus_clustering(cluster_labels_list):
    """
    Combine cluster assignments from multiple clustering algorithms to form a consensus.

    The consensus is formed by taking the mode (most common cluster label)
    among the arrays of labels for each data point. If ties occur or a point is noise
    (-1), assign a new unique label.

    Parameters
    ----------
    cluster_labels_list : list of np.ndarray
        A list of label arrays from different clustering algorithms, all of the same length.

    Returns
    -------
    np.ndarray
        The consensus cluster labels.
    """
    from scipy.stats import mode
    labels_array = np.array(cluster_labels_list).T
    consensus_labels, counts = mode(labels_array, axis=1, nan_policy='omit')
    consensus_labels = consensus_labels.flatten()

    # Handle cases where there is no consensus or all noise
    max_label = consensus_labels.max() + 1
    for idx, (label, count) in enumerate(zip(consensus_labels, counts.flatten())):
        if count == 1 or label == -1:
            consensus_labels[idx] = max_label
            max_label += 1
    return consensus_labels.astype(int)

def relabel_clusters(labels):
    """
    Relabel clusters so that they form a consecutive integer sequence starting from 0.

    Parameters
    ----------
    labels : np.ndarray
        Array of labels that may have arbitrary integers.

    Returns
    -------
    np.ndarray
        Labels remapped to consecutive integers.
    """
    unique_labels = np.unique(labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    new_labels = np.array([label_mapping[label] for label in labels])
    return new_labels


def cluster(vectorized_logs, min_samples=5):
    """
    Perform clustering using DBSCAN, HDBSCAN, and OPTICS, then do consensus clustering.

    Steps:
    1. Optimize eps for DBSCAN using silhouette-based search.
    2. Run DBSCAN with the chosen eps.
    3. Run HDBSCAN and OPTICS.
    4. Combine results using consensus clustering.
    5. Relabel clusters and return.

    Parameters
    ----------
    vectorized_logs : sparse or dense array
        The vectorized log data.
    min_samples : int, optional
        Minimum samples for DBSCAN and other density-based methods, by default 5.

    Returns
    -------
    tuple
        (labels, cluster_nums) where labels are the consensus cluster labels and cluster_nums is the number of clusters formed.
    """
    # Use silhouette-based optimization for DBSCAN eps
    best_eps = optimize_eps_for_dbscan(vectorized_logs, min_samples=min_samples)
    labels_dbscan = dbscan_clustering(vectorized_logs, best_eps, min_samples=min_samples)
    labels_hdbscan = hdbscan_clustering(vectorized_logs, min_cluster_size=min_samples)
    labels_optics = optics_clustering(vectorized_logs, min_samples=min_samples)

    # Perform consensus clustering
    labels = consensus_clustering([labels_dbscan, labels_hdbscan, labels_optics])
    labels = relabel_clusters(labels)
    cluster_nums = len(np.unique(labels))
    return labels, cluster_nums
    

def reassign_clusters(labels, cluster_nums, tokenized_logs):
    """
    Reassign clusters to handle noise points that are identical. 
    Points that were noise (-1) and share the same textual log content
    get grouped into new unique cluster IDs.

    After reassigning, relabel clusters to ensure a consecutive numbering.

    Parameters
    ----------
    labels : np.ndarray
        Current cluster labels.
    cluster_nums : int
        Number of clusters currently known.
    tokenized_logs : list of list of str
        The tokenized logs corresponding to the labels.

    Returns
    -------
    tuple
        (labels, cluster_nums) after reassigning and relabeling.
    """
    merged_logs = []
    for tokenized_log in tokenized_logs:
        merged_logs.append(' '.join(tokenized_log))

    for i in range(len(labels)):
        if labels[i] == -1:
            for j in range(i + 1, len(labels)):
                if labels[j] == -1 and merged_logs[i] == merged_logs[j]:
                    labels[j] = cluster_nums
            labels[i] = cluster_nums
            cluster_nums += 1

    # After reassigning clusters, relabel labels to consecutive integers
    labels = relabel_clusters(labels)
    cluster_nums = len(np.unique(labels))
    return labels, cluster_nums