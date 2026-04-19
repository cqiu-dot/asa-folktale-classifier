"""
Classifier module: Logistic regression on sentence transformer embeddings
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.covariance import LedoitWolf
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)


class FolktaleClassifier:
    """Logistic regression classifier for ATU folktale categories"""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        Initialize classifier
        
        Args:
            config: Configuration dictionary
            device: Device to use ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        
        # Load sentence transformer
        model_name = config['model']['sentence_transformer']
        logger.info(f"Loading sentence transformer: {model_name}")
        self.sentence_transformer = SentenceTransformer(model_name, device=device)
        
        # Initialize logistic regression
        self.classifier = LogisticRegression(
            C=config['model']['C'],
            penalty=config['model']['regularization'],
            max_iter=config['model']['max_iter'],
            solver=config['model']['solver'],
            random_state=config['model']['random_state'],
            verbose=1,
            n_jobs=-1
        )
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        # Set after training; used for Mahalanobis OOD detection
        self.class_means_: Optional[np.ndarray] = None
        self.precision_matrix_: Optional[np.ndarray] = None
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using sentence transformer
        
        Args:
            texts: List of text strings
            
        Returns:
            (n_texts, embedding_dim) array of embeddings
        """
        logger.info(f"Encoding {len(texts)} texts to embeddings")
        embeddings = self.sentence_transformer.encode(
            texts,
            batch_size=self.config['training']['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def train(self, 
              texts: List[str], 
              labels: List[str],
              validation_texts: List[str] = None,
              validation_labels: List[str] = None) -> Dict:
        """
        Train logistic regression classifier on embeddings
        
        Args:
            texts: List of training texts
            labels: List of ATU category labels
            validation_texts: Optional validation texts
            validation_labels: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training on {len(texts)} texts")
        
        # Encode texts
        X = self.encode_texts(texts)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Training logistic regression with {X.shape[1]} features")
        self.classifier.fit(X, y)
        self.is_fitted = True

        # Compute class centroids and shared precision matrix for Mahalanobis OOD detection
        n_classes = len(self.label_encoder.classes_)
        self.class_means_ = np.array([X[y == k].mean(axis=0) for k in range(n_classes)])
        # Stack within-class residuals and fit a shrinkage covariance estimate
        centered = np.vstack([X[y == k] - self.class_means_[k] for k in range(n_classes)])
        lw = LedoitWolf()
        lw.fit(centered)
        self.precision_matrix_ = lw.precision_

        # Compute training metrics
        train_pred = self.classifier.predict(X)
        train_acc = accuracy_score(y, train_pred)
        train_f1 = f1_score(y, train_pred, average='weighted')
        
        metrics = {
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        
        # Validation metrics if provided
        if validation_texts is not None and validation_labels is not None:
            X_val = self.encode_texts(validation_texts)
            y_val = self.label_encoder.transform(validation_labels)
            
            val_pred = self.classifier.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            metrics['val_accuracy'] = val_acc
            metrics['val_f1'] = val_f1
        
        logger.info(f"Training complete. Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        return metrics
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict ATU categories for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            (predictions, probabilities) tuple
            - predictions: (n_texts,) array of predicted class indices
            - probabilities: (n_texts, n_classes) array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before prediction")
        
        X = self.encode_texts(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities
    
    def get_confidence_metrics(self, probabilities: np.ndarray) -> Dict:
        """
        Compute confidence metrics from predictions
        
        Args:
            probabilities: (n_texts, n_classes) probability matrix
            
        Returns:
            Dictionary with confidence metrics
        """
        n_classes = probabilities.shape[1]
        max_probs = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        # Normalize entropy to [0, 1]: 0 = certain, 1 = uniform/random-guess
        normalized_entropy = entropy / np.log(n_classes)
        # Margin: gap between top-1 and top-2 probability
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        return {
            'max_confidence': max_probs,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'margin': margin,
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'mean_entropy': np.mean(entropy),
            'std_entropy': np.std(entropy),
            'mean_normalized_entropy': np.mean(normalized_entropy),
            'std_normalized_entropy': np.std(normalized_entropy),
            'mean_margin': np.mean(margin),
            'std_margin': np.std(margin),
        }
    
    def mahalanobis_distances(self, texts: List[str]) -> np.ndarray:
        """
        Compute minimum Mahalanobis distance from each text to any training class centroid.

        Large distance means the input is far from the training distribution (OOD).
        This measure is independent of softmax calibration.

        Args:
            texts: List of text strings

        Returns:
            (n_texts,) array of minimum Mahalanobis distances
        """
        if self.class_means_ is None or self.precision_matrix_ is None:
            raise ValueError("Model must be trained before computing Mahalanobis distances")

        X = self.encode_texts(texts)
        # Compute distance from each point to each class centroid, take minimum
        dists = np.zeros((len(X), len(self.class_means_)))
        for k, mu in enumerate(self.class_means_):
            diff = X - mu  # (n, d)
            # (x - mu)^T Σ^{-1} (x - mu), computed row-wise
            dists[:, k] = np.sqrt(np.einsum('ij,jk,ik->i', diff, self.precision_matrix_, diff))
        return np.min(dists, axis=1)

    def save(self, filepath: str):
        """Save trained model and preprocessing objects"""
        objects = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'class_means': self.class_means_,
            'precision_matrix': self.precision_matrix_,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load trained model and preprocessing objects"""
        with open(filepath, 'rb') as f:
            objects = pickle.load(f)
        self.classifier = objects['classifier']
        self.label_encoder = objects['label_encoder']
        self.config = objects['config']
        self.class_means_ = objects.get('class_means')
        self.precision_matrix_ = objects.get('precision_matrix')
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    # config = yaml.safe_load(open('config/config.yaml'))
    # clf = FolktaleClassifier(config)
    # metrics = clf.train(texts, labels)
