"""
Classifier module: Logistic regression on sentence transformer embeddings
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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
            regularization=config['model']['regularization'],
            max_iter=config['model']['max_iter'],
            solver=config['model']['solver'],
            random_state=config['model']['random_state'],
            verbose=1,
            n_jobs=-1
        )
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
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
        max_probs = np.max(probabilities, axis=1)  # Maximum probability
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)  # Shannon entropy
        
        return {
            'max_confidence': max_probs,
            'entropy': entropy,
            'mean_confidence': np.mean(max_probs),
            'mean_entropy': np.mean(entropy),
            'std_confidence': np.std(max_probs),
            'std_entropy': np.std(entropy)
        }
    
    def save(self, filepath: str):
        """Save trained model and preprocessing objects"""
        objects = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'config': self.config
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
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    # config = yaml.safe_load(open('config/config.yaml'))
    # clf = FolktaleClassifier(config)
    # metrics = clf.train(texts, labels)
