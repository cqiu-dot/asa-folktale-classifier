"""
Visualization utilities for folktale classifier results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def plot_confidence_distribution(confidences: np.ndarray,
                                entropies: np.ndarray,
                                label: str = "Folktales",
                                output_path: Optional[str] = None):
    """
    Plot confidence and entropy distributions
    
    Args:
        confidences: Array of max probabilities
        entropies: Array of Shannon entropies
        label: Label for the plot
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence distribution
    axes[0].hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
    axes[0].set_xlabel('Max Probability (Confidence)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{label}: Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Entropy distribution
    axes[1].hist(entropies, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(entropies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(entropies):.3f}')
    axes[1].set_xlabel('Shannon Entropy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{label}: Entropy Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")
    
    plt.show()


def plot_confidence_comparison(western_confidences: np.ndarray,
                               asian_confidences: np.ndarray,
                               output_path: Optional[str] = None):
    """
    Compare confidence distributions between Western and Asian folktales
    
    Args:
        western_confidences: Array of confidences for Western tales
        asian_confidences: Array of confidences for Asian tales
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(western_confidences, bins=30, alpha=0.6, label='Western (ATU-labeled)',
           color='blue', edgecolor='black')
    ax.hist(asian_confidences, bins=30, alpha=0.6, label='Asian (unlabeled)',
           color='red', edgecolor='black')
    
    ax.axvline(np.mean(western_confidences), color='blue', linestyle='--',
              linewidth=2, label=f'Western Mean: {np.mean(western_confidences):.3f}')
    ax.axvline(np.mean(asian_confidences), color='red', linestyle='--',
              linewidth=2, label=f'Asian Mean: {np.mean(asian_confidences):.3f}')
    
    ax.set_xlabel('Max Probability (Confidence)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Classifier Confidence: Western vs. Asian Folktales', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison figure to {output_path}")
    
    plt.show()


def plot_cluster_sizes(cluster_indices: np.ndarray,
                      output_path: Optional[str] = None):
    """
    Plot cluster size distribution
    
    Args:
        cluster_indices: Array of cluster assignments
        output_path: Path to save figure
    """
    unique, counts = np.unique(cluster_indices, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Stories', fontsize=12)
    ax.set_title('Distribution of Stories Across Clusters', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (cluster_id, count) in enumerate(zip(unique, counts)):
        ax.text(cluster_id, count + 1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster plot to {output_path}")
    
    plt.show()


def plot_tsne_clusters(embeddings: np.ndarray,
                      cluster_indices: np.ndarray,
                      labels: Optional[np.ndarray] = None,
                      output_path: Optional[str] = None):
    """
    Plot 2D t-SNE visualization of clusters
    
    Args:
        embeddings: (n, d) embedding matrix
        cluster_indices: Cluster assignments
        labels: Optional class labels for coloring
        output_path: Path to save figure
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn TSNE not available. Skipping visualization.")
        return
    
    logger.info("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(cluster_indices))))
    for cluster_id in np.unique(cluster_indices):
        mask = cluster_indices == cluster_id
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                  s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Folktale Clusters', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved t-SNE plot to {output_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str],
                         output_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
