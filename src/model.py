"""
Classifier module for ATU folktale category classification.

Architecture:
  - Features: Word TF-IDF (1-3gram, 50k) + Char TF-IDF (3-5gram, 30k)
               + L2-normalised sentence-transformer embeddings (384-dim)
  - Classifier: LinearSVC wrapped in CalibratedClassifierCV (Platt scaling)
                Gives both good discrimination and calibrated probabilities.
  - OOD detector: Mahalanobis distance in the pure 384-dim embedding space
                  (TF-IDF space is too sparse/high-dim for Mahalanobis)

ATU grouping: 14 scholarly-aligned classes from Uther (2004) named subdivisions.
  Raw ATU IDs (160 classes, ~8 examples each): ~25% accuracy
  7 broad categories: ~79% accuracy but too coarse for narrative analysis
  14 scholarly classes: ~71% CV accuracy on 1484 examples; monotonically
  increasing with data -- the model is data-limited, not capacity-limited.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import LedoitWolf
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)

# ── ATU 14-class scholarly grouping ─────────────────────────────────────────
# Based on Uther (2004) named subdivisions. Splits the monolithic "Tales of
# Magic" block (300-749, 518 tales) into six semantically distinct subgroups,
# and divides Anecdotes & Jokes (1200-1999) into three thematic bands.
# Each class has 46-246 examples (mean ~106, min 46).

ATU_LABELS = {
    'animal_small':       'Animal Tales: Birds, Fish, Invertebrates (ATU 1-99)',
    'animal_wild':        'Animal Tales: Wild & Domestic Animals (ATU 100-299)',
    'magic_adversary':    'Tales of Magic: Supernatural Adversary (ATU 300-399)',
    'magic_spouse':       'Tales of Magic: Enchanted Spouse / Tasks (ATU 400-499)',
    'magic_helper':       'Tales of Magic: Supernatural Helpers (ATU 500-559)',
    'magic_object':       'Tales of Magic: Magic Objects (ATU 560-649)',
    'magic_power':        'Tales of Magic: Special Power / Other Supernatural (ATU 650-749)',
    'religious':          'Religious Tales (ATU 750-849)',
    'realistic':          'Realistic / Novelistic Tales (ATU 850-999)',
    'ogre':               'Tales of the Stupid Ogre (ATU 1000-1199)',
    'jokes_wise_foolish': 'Anecdotes: Wise & Foolish Men (ATU 1200-1399)',
    'jokes_domestic':     'Anecdotes: Married Couples & Women (ATU 1400-1599)',
    'jokes_clever':       'Anecdotes: Clever Men & Lucky Accidents (ATU 1600-1999)',
    'formula':            'Formula Tales (ATU 2000+)',
}

def atu_scholarly_category(atu_id: str) -> str:
    """
    Map a raw ATU ID string (e.g. '510A', '1645') to a 14-class scholarly key.
    Returns None for unparseable IDs.
    """
    m = re.match(r'(\d+)', str(atu_id))
    if not m:
        return None
    n = int(m.group(1))
    if n <  100: return 'animal_small'
    if n <  300: return 'animal_wild'
    if n <  400: return 'magic_adversary'
    if n <  500: return 'magic_spouse'
    if n <  560: return 'magic_helper'
    if n <  650: return 'magic_object'
    if n <  750: return 'magic_power'
    if n <  850: return 'religious'
    if n < 1000: return 'realistic'
    if n < 1200: return 'ogre'
    if n < 1400: return 'jokes_wise_foolish'
    if n < 1600: return 'jokes_domestic'
    if n < 2000: return 'jokes_clever'
    return 'formula'

# Keep the 7-class version as a convenience alias for backward compatibility
def atu_major_category(atu_id: str) -> str:
    """Coarser 7-class grouping. Prefer atu_scholarly_category for new code."""
    m = re.match(r'(\d+)', str(atu_id))
    if not m:
        return 'unknown'
    n = int(m.group(1))
    if n < 300:   return 'animal'
    if n < 750:   return 'magic'
    if n < 850:   return 'religious'
    if n < 1000:  return 'realistic'
    if n < 1200:  return 'ogre'
    if n < 2000:  return 'anecdotes'
    return 'formula'


class FolktaleClassifier:
    """
    ATU folktale classifier.

    Features: Word TF-IDF (1-3gram) + Char TF-IDF (3-5gram) + sentence embeddings
    Classifier: LinearSVC + CalibratedClassifierCV (Platt scaling for probabilities)
    OOD detection: Mahalanobis distance in the 384-dim pure embedding space
    """

    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.use_tfidf: bool = config.get('model', {}).get('use_tfidf', True)

        model_name = config['model']['sentence_transformer']
        logger.info(f"Loading sentence transformer: {model_name}")
        self.sentence_transformer = SentenceTransformer(model_name, device=device)

        # LinearSVC wrapped in Platt-scaling calibration for probability estimates.
        # LinearSVC consistently outperforms logistic regression on TF-IDF text features
        # (~79% vs ~77% on 7-class, ~71% vs ~72% on 14-class in CV).
        C = config['model']['C']
        self.classifier = CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=config['model']['max_iter'],
                      random_state=config['model']['random_state']),
            method='sigmoid',  # Platt scaling
            cv=5,
        )

        # TF-IDF vectorizers (fitted during train())
        self.tfidf_word: Optional[TfidfVectorizer] = (
            TfidfVectorizer(ngram_range=(1, 3), max_features=50000,
                            sublinear_tf=True, min_df=2)
            if self.use_tfidf else None
        )
        self.tfidf_char: Optional[TfidfVectorizer] = (
            TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                            max_features=30000, sublinear_tf=True, min_df=3)
            if self.use_tfidf else None
        )
        # Legacy alias kept for load() backward compatibility
        self.tfidf = self.tfidf_word

        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.class_means_: Optional[np.ndarray] = None   # in embedding space only
        self.precision_matrix_: Optional[np.ndarray] = None
    
    # ── Feature construction ─────────────────────────────────────────────────

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Return (n, 384) sentence-transformer embeddings for texts."""
        logger.info(f"Encoding {len(texts)} texts to embeddings")
        return self.sentence_transformer.encode(
            texts,
            batch_size=self.config['training']['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def _build_features(self, texts: List[str], fit_tfidf: bool = False):
        """
        Build classifier input features, returning (X_clf, X_emb).

        X_clf  — features for LinearSVC: [word-TF-IDF | char-TF-IDF | L2-emb]
        X_emb  — raw sentence-transformer embeddings (384-dim, for Mahalanobis)

        Mahalanobis OOD detection always uses X_emb, not X_clf.
        """
        X_emb = self.encode_texts(texts)

        if not self.use_tfidf:
            return X_emb, X_emb

        if fit_tfidf:
            X_word = self.tfidf_word.fit_transform(texts)
            X_char = self.tfidf_char.fit_transform(texts)
            self.tfidf = self.tfidf_word  # keep legacy alias in sync
        else:
            X_word = self.tfidf_word.transform(texts)
            X_char = self.tfidf_char.transform(texts)

        X_emb_normed = csr_matrix(normalize(X_emb))
        X_combined = hstack([X_word, X_char, X_emb_normed])
        return X_combined, X_emb

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self,
              texts: List[str],
              labels: List[str],
              validation_texts: List[str] = None,
              validation_labels: List[str] = None) -> Dict:
        """Train the classifier. Labels should be major ATU category keys."""
        logger.info(f"Training on {len(texts)} texts  (use_tfidf={self.use_tfidf})")

        X, X_emb = self._build_features(texts, fit_tfidf=True)
        y = self.label_encoder.fit_transform(labels)

        n_feat = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        logger.info(f"Fitting logistic regression on {n_feat} features, {len(set(labels))} classes")
        self.classifier.fit(X, y)
        self.is_fitted = True

        # Mahalanobis OOD detection: class centroids in the pure embedding space
        n_classes = len(self.label_encoder.classes_)
        self.class_means_ = np.array([X_emb[y == k].mean(axis=0) for k in range(n_classes)])
        centered = np.vstack([X_emb[y == k] - self.class_means_[k] for k in range(n_classes)])
        lw = LedoitWolf()
        lw.fit(centered)
        self.precision_matrix_ = lw.precision_

        train_pred = self.classifier.predict(X)
        train_acc = accuracy_score(y, train_pred)
        train_f1  = f1_score(y, train_pred, average='weighted')

        metrics = {
            'train_accuracy': train_acc,
            'train_f1':       train_f1,
            'n_features':     n_feat,
            'n_classes':      n_classes,
            'classes':        list(self.label_encoder.classes_),
            'use_tfidf':      self.use_tfidf,
        }

        if validation_texts is not None and validation_labels is not None:
            X_val, _ = self._build_features(validation_texts, fit_tfidf=False)
            y_val    = self.label_encoder.transform(validation_labels)
            val_pred = self.classifier.predict(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
            metrics['val_f1']       = f1_score(y_val, val_pred, average='weighted')

        logger.info(f"Training complete. Train acc={train_acc:.4f}  F1={train_f1:.4f}")
        return metrics

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (predictions, probabilities) for texts.

        predictions:   (n,) int array of predicted class indices
        probabilities: (n, n_classes) softmax probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before prediction")

        X, _ = self._build_features(texts, fit_tfidf=False)
        predictions   = self.classifier.predict(X)
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
    
    def evaluate_indistribution(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Full in-distribution evaluation on the held-out Western test set.

        Returns per-class classification report, macro F1, weighted F1, accuracy,
        and Expected Calibration Error (ECE). Call this before any OOD testing —
        the OOD confidence-gap argument only holds if the model learned the ATU
        structure meaningfully.

        ECE measures softmax calibration: 0.0 = perfectly calibrated, higher = worse.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before evaluation")

        X, _ = self._build_features(texts, fit_tfidf=False)
        y_true = self.label_encoder.transform(labels)
        probs  = self.classifier.predict_proba(X)
        y_pred = probs.argmax(axis=1)

        macro_f1    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        acc         = accuracy_score(y_true, y_pred)
        report      = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )

        # ECE: bucket predictions by max-softmax confidence, compare to actual accuracy
        max_probs = probs.max(axis=1)
        correct   = (y_pred == y_true).astype(float)
        n_bins    = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (max_probs >= lo) & (max_probs < hi)
            if mask.sum() > 0:
                ece += mask.sum() * abs(correct[mask].mean() - max_probs[mask].mean())
        ece /= len(y_true)

        random_baseline = 1.0 / len(self.label_encoder.classes_)

        return {
            'accuracy':         acc,
            'macro_f1':         macro_f1,
            'weighted_f1':      weighted_f1,
            'ece':              ece,
            'random_baseline':  random_baseline,
            'n_samples':        len(y_true),
            'n_classes':        len(self.label_encoder.classes_),
            'classification_report': report,
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

    def cross_validated_confidence(
        self,
        texts: List[str],
        labels: List[str],
        n_splits: int = 5,
    ) -> Dict:
        """
        5-fold cross-validated confidence estimation on Western tales.

        Each Western tale is held out exactly once: the model trains on 4 folds,
        then assigns confidence scores to the 5th fold without ever having seen
        those examples. Pooling across all folds gives an unbiased in-distribution
        confidence baseline.

        This is the rigorous baseline for the OOD comparison: every Western tale
        contributes, and no confidence is ever measured on training data.

        Returns a dict with:
          probs            (n, K)  pooled held-out probabilities (original row order)
          mh_distances     (n,)    pooled Mahalanobis distances
          cv_accuracies    list of per-fold held-out accuracy
          cv_macro_f1s     list of per-fold macro F1
          mean_cv_accuracy mean of cv_accuracies
          mean_cv_macro_f1 mean of cv_macro_f1s
          labels_encoded   (n,) ground-truth encoded labels in original order
        """
        # Pre-encode all texts once — expensive step, do it outside the CV loop
        print("  Pre-encoding all texts for cross-validation …", flush=True)
        all_emb = self.encode_texts(texts)

        # Encode labels to integers using the full label set
        le_full = LabelEncoder().fit(labels)
        y_all   = le_full.transform(labels)
        n       = len(texts)
        K       = len(le_full.classes_)

        probs_pooled = np.zeros((n, K), dtype=np.float32)
        mh_pooled    = np.zeros(n, dtype=np.float32)
        cv_accuracies, cv_macro_f1s = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, y_all)):
            print(f"  Fold {fold_idx+1}/{n_splits} …", flush=True)

            train_texts = [texts[i] for i in train_idx]
            val_texts   = [texts[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels   = [labels[i] for i in val_idx]
            X_emb_train = all_emb[train_idx]
            X_emb_val   = all_emb[val_idx]
            y_train      = y_all[train_idx]
            y_val        = y_all[val_idx]

            # Fit fresh TF-IDF on training fold only
            tfidf_w = TfidfVectorizer(ngram_range=(1, 3), max_features=50000,
                                       sublinear_tf=True, min_df=2)
            tfidf_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                       max_features=30000, sublinear_tf=True, min_df=3)
            X_word_tr = tfidf_w.fit_transform(train_texts)
            X_char_tr = tfidf_c.fit_transform(train_texts)
            X_emb_tr_normed = csr_matrix(normalize(X_emb_train))
            X_train_clf = hstack([X_word_tr, X_char_tr, X_emb_tr_normed])

            X_word_v = tfidf_w.transform(val_texts)
            X_char_v = tfidf_c.transform(val_texts)
            X_emb_v_normed = csr_matrix(normalize(X_emb_val))
            X_val_clf = hstack([X_word_v, X_char_v, X_emb_v_normed])

            C   = self.config['model']['C']
            clf = CalibratedClassifierCV(
                LinearSVC(C=C, max_iter=self.config['model']['max_iter'],
                          random_state=self.config['model']['random_state']),
                method='sigmoid', cv=5,
            )
            clf.fit(X_train_clf, y_train)

            fold_probs = clf.predict_proba(X_val_clf)
            fold_preds = fold_probs.argmax(axis=1)
            cv_accuracies.append(accuracy_score(y_val, fold_preds))
            cv_macro_f1s.append(f1_score(y_val, fold_preds, average='macro', zero_division=0))

            # Store probabilities mapped to the full K-class label space
            fold_classes = clf.classes_
            for col_in_fold, global_class in enumerate(fold_classes):
                probs_pooled[val_idx, global_class] += fold_probs[:, col_in_fold]

            # Mahalanobis: fit on training fold embeddings
            class_means = np.array([X_emb_train[y_train == k].mean(axis=0)
                                     for k in range(K)
                                     if (y_train == k).any()])
            centered = np.vstack([X_emb_train[y_train == k] - X_emb_train[y_train == k].mean(axis=0)
                                   for k in range(K) if (y_train == k).any()])
            lw = LedoitWolf()
            lw.fit(centered)
            prec = lw.precision_
            dists_val = np.zeros((len(val_idx), len(class_means)))
            for k, mu in enumerate(class_means):
                diff = X_emb_val - mu
                dists_val[:, k] = np.sqrt(np.einsum('ij,jk,ik->i', diff, prec, diff))
            mh_pooled[val_idx] = dists_val.min(axis=1)

        return {
            'probs':            probs_pooled,
            'mh_distances':     mh_pooled,
            'cv_accuracies':    cv_accuracies,
            'cv_macro_f1s':     cv_macro_f1s,
            'mean_cv_accuracy': float(np.mean(cv_accuracies)),
            'mean_cv_macro_f1': float(np.mean(cv_macro_f1s)),
            'labels_encoded':   y_all,
            'label_classes':    list(le_full.classes_),
        }

    def save(self, filepath: str):
        """Save trained model and all preprocessing objects."""
        objects = {
            'classifier':       self.classifier,
            'label_encoder':    self.label_encoder,
            'tfidf_word':       self.tfidf_word,
            'tfidf_char':       self.tfidf_char,
            'config':           self.config,
            'class_means':      self.class_means_,
            'precision_matrix': self.precision_matrix_,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load trained model and all preprocessing objects."""
        with open(filepath, 'rb') as f:
            objects = pickle.load(f)
        self.classifier        = objects['classifier']
        self.label_encoder     = objects['label_encoder']
        self.tfidf_word        = objects.get('tfidf_word') or objects.get('tfidf')
        self.tfidf_char        = objects.get('tfidf_char')
        self.tfidf             = self.tfidf_word
        self.use_tfidf         = self.tfidf_word is not None
        self.config            = objects['config']
        self.class_means_      = objects.get('class_means')
        self.precision_matrix_ = objects.get('precision_matrix')
        self.is_fitted         = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    # config = yaml.safe_load(open('config/config.yaml'))
    # clf = FolktaleClassifier(config)
    # metrics = clf.train(texts, labels)
