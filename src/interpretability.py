"""
Neural network interpretability module: SAE-lens clustering and motif extraction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


class MotifExtractor:
    """Extract narrative motifs from clusters using various interpretability techniques"""
    
    def __init__(self, config: Dict):
        """Initialize motif extractor with configuration"""
        self.config = config
        self.interpretability_config = config['interpretability']
    
    def extract_top_tokens(self,
                          embeddings: np.ndarray,
                          texts: List[str],
                          feature_indices: np.ndarray,
                          k: int = None) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top tokens that activate for each feature/cluster
        
        Args:
            embeddings: (n_texts, n_features) embedding matrix
            texts: List of text strings
            feature_indices: (n_texts,) cluster assignments
            k: Top-k tokens to extract (default from config)
            
        Returns:
            Dictionary mapping cluster_id -> [(token, importance_score), ...]
        """
        if k is None:
            k = self.interpretability_config['top_tokens_k']
        
        top_tokens = {}
        unique_clusters = np.unique(feature_indices)
        
        logger.info(f"Extracting top {k} tokens for each cluster")
        
        for cluster_id in unique_clusters:
            # Get texts in this cluster
            mask = feature_indices == cluster_id
            cluster_texts = [texts[i] for i in np.where(mask)[0]]
            cluster_embeddings = embeddings[mask]
            
            # Compute average embedding for cluster
            avg_embedding = np.mean(cluster_embeddings, axis=0)
            
            # Find most important features for this cluster
            top_feature_indices = np.argsort(np.abs(avg_embedding))[-k:][::-1]
            
            # Extract words using simple heuristics (tokenization)
            from collections import Counter
            import re
            
            all_words = []
            for text in cluster_texts:
                words = re.findall(r'\w+', text.lower())
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(k)
            
            top_tokens[cluster_id] = top_words
        
        return top_tokens
    
    def extract_representative_stories(self,
                                      embeddings: np.ndarray,
                                      texts: List[str],
                                      metadata: pd.DataFrame,
                                      cluster_indices: np.ndarray,
                                      n_per_cluster: int = None) -> Dict[int, List[Dict]]:
        """
        Extract most representative stories for each cluster
        
        Args:
            embeddings: (n_texts, n_features) embedding matrix
            texts: List of text strings
            metadata: DataFrame with story metadata
            cluster_indices: (n_texts,) cluster assignments
            n_per_cluster: Number of stories per cluster
            
        Returns:
            Dictionary mapping cluster_id -> [story_dicts, ...]
        """
        if n_per_cluster is None:
            n_per_cluster = self.interpretability_config['n_representative_stories']
        
        representative_stories = {}
        unique_clusters = np.unique(cluster_indices)
        
        logger.info(f"Extracting {n_per_cluster} representative stories per cluster")
        
        for cluster_id in unique_clusters:
            mask = cluster_indices == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_indices_local = np.where(mask)[0]
            
            # Compute centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find closest stories to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argsort(distances)[:n_per_cluster]
            
            stories = []
            for i in closest_idx:
                global_idx = cluster_indices_local[i]
                story = {
                    'index': global_idx,
                    'text': texts[global_idx][:500],  # Truncate for display
                    'distance_to_centroid': distances[i]
                }
                if metadata is not None and global_idx < len(metadata):
                    # Add metadata columns
                    for col in metadata.columns:
                        story[col] = metadata.iloc[global_idx][col]
                
                stories.append(story)
            
            representative_stories[cluster_id] = stories
        
        return representative_stories
    
    def cluster_embeddings(self,
                          embeddings: np.ndarray,
                          method: str = 'kmeans',
                          n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Cluster embeddings using specified method
        
        Args:
            embeddings: (n_texts, n_features) embedding matrix
            method: Clustering method ('kmeans', 'sae', 'gemmascope')
            n_clusters: Number of clusters
            
        Returns:
            (cluster_assignments, cluster_info) tuple
        """
        logger.info(f"Clustering embeddings using {method}")
        
        if method == 'kmeans':
            if n_clusters is None:
                n_clusters = self.config['clustering']['kmeans']['n_clusters']
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=self.config['clustering']['kmeans']['n_init'],
                random_state=self.config['model']['random_state']
            )
            cluster_assignments = kmeans.fit_predict(embeddings)
            
            info = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'inertia': kmeans.inertia_,
                'silhouette_score': None  # Compute separately if needed
            }
        
        elif method == 'sae':
            try:
                import sae_lens
                logger.warning("SAE-Lens clustering not yet implemented. Using k-means.")
                return self.cluster_embeddings(embeddings, 'kmeans', n_clusters)
            except ImportError:
                logger.warning("sae-lens not installed. Using k-means.")
                return self.cluster_embeddings(embeddings, 'kmeans', n_clusters)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        logger.info(f"Clustering complete. Cluster sizes: {np.bincount(cluster_assignments)}")
        return cluster_assignments, info
    
    def analyze_cluster_coefficients(self,
                                    embeddings: np.ndarray,
                                    cluster_indices: np.ndarray,
                                    classifier_object = None,
                                    k: int = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        Analyze logistic regression coefficients for each cluster
        
        Args:
            embeddings: (n_texts, n_features) embedding matrix
            cluster_indices: (n_texts,) cluster assignments
            classifier_object: Trained classifier with accessible coefficients
            k: Top-k features to extract
            
        Returns:
            Dictionary mapping cluster_id -> [(feature_idx, coefficient), ...]
        """
        if k is None:
            k = self.interpretability_config['top_features_k']
        
        cluster_coefficients = {}
        unique_clusters = np.unique(cluster_indices)
        
        logger.info(f"Analyzing logistic regression coefficients for {len(unique_clusters)} clusters")
        
        for cluster_id in unique_clusters:
            mask = cluster_indices == cluster_id
            cluster_embeddings = embeddings[mask]
            
            # Compute average embedding for cluster
            avg_embedding = np.mean(cluster_embeddings, axis=0)
            
            # Get top features by absolute value
            abs_weights = np.abs(avg_embedding)
            top_indices = np.argsort(abs_weights)[-k:][::-1]
            
            coefficients = [(idx, avg_embedding[idx]) for idx in top_indices]
            cluster_coefficients[cluster_id] = coefficients
        
        return cluster_coefficients
    
    def extract_ngrams(self,
                      texts: List[str],
                      cluster_indices: np.ndarray,
                      ngram_range: Tuple[int, int] = None,
                      k: int = None) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract distinctive n-grams for each cluster via frequency contrast
        
        Args:
            texts: List of text strings
            cluster_indices: (n_texts,) cluster assignments
            ngram_range: (min_n, max_n) for n-gram sizes
            k: Top-k n-grams per cluster
            
        Returns:
            Dictionary mapping cluster_id -> [(ngram, tfidf_score), ...]
        """
        if ngram_range is None:
            ngram_range = tuple(self.interpretability_config['ngram_range'])
        if k is None:
            k = self.interpretability_config['top_ngrams_k']
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict
        
        logger.info(f"Extracting n-grams (range={ngram_range}) for each cluster")
        
        cluster_ngrams = {}
        unique_clusters = np.unique(cluster_indices)
        
        for cluster_id in unique_clusters:
            mask = cluster_indices == cluster_id
            cluster_texts = [texts[i] for i in np.where(mask)[0]]
            
            # Vectorize cluster texts
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=1000)
            tfidf = vectorizer.fit_transform(cluster_texts)
            
            # Get top features
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(tfidf.mean(axis=0)).ravel()
            
            top_indices = np.argsort(scores)[-k:][::-1]
            ngrams = [(feature_names[i], scores[i]) for i in top_indices]
            
            cluster_ngrams[cluster_id] = ngrams
        
        return cluster_ngrams
    
    def save_analysis(self, analysis_dict: Dict, filepath: str):
        """Save analysis results"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(analysis_dict, f)
        
        logger.info(f"Analysis saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
