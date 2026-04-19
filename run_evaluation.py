"""
Standalone evaluation: train ATU classifier on Western tales, test confidence on Asian tales.
Run from the project root: py run_evaluation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yaml
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats

from src.data_loader import FolktaleDataLoader
from src.model import FolktaleClassifier

logging.basicConfig(level=logging.WARNING)   # suppress info noise for cleaner output

# ── Config ──────────────────────────────────────────────────────────────────
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# ── Western tales ────────────────────────────────────────────────────────────
print("Loading Western folktales …")
western_df = pd.read_csv('../aft.csv')
western_df['text_processed'] = western_df['text'].fillna('').apply(
    lambda t: ' '.join(t.split()))

# Use atu_id as label; filter rare classes so every class has ≥3 examples
label_counts = western_df['atu_id'].value_counts()
keep = label_counts[label_counts >= 3].index
western_df = western_df[western_df['atu_id'].isin(keep)].reset_index(drop=True)

texts  = western_df['text_processed'].values
labels = western_df['atu_id'].values
print(f"  {len(texts)} tales, {len(set(labels))} ATU categories (each >=3 examples)")

texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train, labels_train, test_size=0.1, random_state=42)

# ── Train ────────────────────────────────────────────────────────────────────
print("\nTraining classifier …")
clf = FolktaleClassifier(config, device='cpu')
metrics = clf.train(texts_train.tolist(), labels_train.tolist(),
                    texts_val.tolist(), labels_val.tolist())
print(f"  Train acc={metrics['train_accuracy']:.3f}  "
      f"Val acc={metrics.get('val_accuracy', float('nan')):.3f}  "
      f"n_classes={metrics['n_classes']}")

# ── Western test-set confidence (baseline) ───────────────────────────────────
print("\nEvaluating on Western held-out test set …")
_, probs_west = clf.predict(texts_test.tolist())
m_west  = clf.get_confidence_metrics(probs_west)
mh_west = clf.mahalanobis_distances(texts_test.tolist())

K = probs_west.shape[1]
print(f"  n_classes K={K}   max possible entropy = log({K}) = {np.log(K):.3f} nats")

# ── Asian tales ──────────────────────────────────────────────────────────────
print("\nLoading Asian/SE Asian folktales …")
asian_records = []
for json_path in [
    '../china_china_fables_dataset (1).json',
    '../japan_japan_fairy_tales_dataset.json',
    '../korea_korea_fairy_tales_dataset.json',
]:
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    region = Path(json_path).stem.split('_')[1]          # china / japan / korea
    for rec in data:
        body = rec.get('body') or rec.get('text') or ''
        if body.strip():
            asian_records.append({'region': region, 'text': body.strip(),
                                  'title': rec.get('title', '')})
asian_df = pd.DataFrame(asian_records)
print(f"  {len(asian_df)} tales  ({asian_df['region'].value_counts().to_dict()})")

texts_asian = asian_df['text'].values.tolist()
_, probs_asian = clf.predict(texts_asian)
m_asian  = clf.get_confidence_metrics(probs_asian)
mh_asian = clf.mahalanobis_distances(texts_asian)

# ── Per-region breakdown ──────────────────────────────────────────────────────
print("\nPer-region normalized entropy:")
for region, grp in asian_df.groupby('region'):
    idx = grp.index.tolist()
    ne  = m_asian['normalized_entropy'][idx].mean()
    print(f"  {region:8s}: mean normalized entropy = {ne:.4f}")

# ── Comparison table ─────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled

print("\n" + "="*90)
print("  Hypothesis test: is the ATU classifier less confident on Asian tales?")
print("="*90)
rows = [
    ("Max-softmax confidence",   m_west['max_confidence'],       m_asian['max_confidence'],      "higher=more confident"),
    ("Normalized entropy H/logK", m_west['normalized_entropy'], m_asian['normalized_entropy'],   "lower=more confident; 1=random"),
    ("Margin p1-p2",              m_west['margin'],               m_asian['margin'],              "higher=more decisive"),
    ("Mahalanobis distance",      mh_west,                        mh_asian,                       "higher=more OOD"),
]
print(f"\n{'Metric':<34} {'Western':>9} {'Asian':>9} {'Delta':>9} {'Cohen d':>9}  {'MW p':>10}  Note")
print("-"*90)
for label, w, a, note in rows:
    delta = np.mean(w) - np.mean(a)
    d     = cohens_d(w, a)
    _, p  = stats.mannwhitneyu(w, a, alternative='two-sided')
    print(f"{label:<34} {np.mean(w):>9.4f} {np.mean(a):>9.4f} {delta:>+9.4f} {d:>9.3f}  {p:>10.2e}  {note}")

# ── Interpretation ────────────────────────────────────────────────────────────
ne_w = m_west['mean_normalized_entropy']
ne_a = m_asian['mean_normalized_entropy']
_, p_ne = stats.mannwhitneyu(m_west['normalized_entropy'],
                              m_asian['normalized_entropy'], alternative='two-sided')
d_ne = cohens_d(m_west['normalized_entropy'], m_asian['normalized_entropy'])

print(f"\n-- Interpretation ------------------------------------------------------------------")
print(f"  Western normalized entropy : {ne_w:.4f}  (baseline: model trained on this distribution)")
print(f"  Asian   normalized entropy : {ne_a:.4f}  (hypothesis: should be higher = more uncertain)")
print(f"  Scale: 0.00 = perfectly certain  |  1.00 = pure random guess over {K} classes")

thresholds = [(0.85, "near-random (ATU does not generalize at all)"),
              (0.65, "highly uncertain (ATU generalizes poorly)"),
              (0.40, "moderately uncertain (partial generalization)"),
              (0.00, "relatively confident (ATU may generalize)")]
for cutoff, label in thresholds:
    if ne_a > cutoff:
        print(f"  Asian H_norm={ne_a:.3f} => {label}")
        break

effect = ("large" if abs(d_ne) >= 0.8 else
          "medium" if abs(d_ne) >= 0.5 else
          "small"  if abs(d_ne) >= 0.2 else "negligible")
print(f"  Gap Western→Asian: Δ={ne_w-ne_a:+.4f}, Cohen's d={d_ne:.3f} ({effect} effect)")
print(f"  Mann-Whitney p={p_ne:.2e}  → hypothesis {'SUPPORTED' if p_ne < 0.05 and ne_a > ne_w else 'NOT supported'} at α=0.05")

# ── Save results ──────────────────────────────────────────────────────────────
Path('results/analysis').mkdir(parents=True, exist_ok=True)
out = {
    'n_atu_classes': K,
    'max_entropy': float(np.log(K)),
    'western': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in m_west.items()},
    'asian':   {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in m_asian.items()},
    'mahalanobis': {'western': mh_west.tolist(), 'asian': mh_asian.tolist()},
    'per_region_normalized_entropy': {
        region: float(m_asian['normalized_entropy'][grp.index].mean())
        for region, grp in asian_df.groupby('region')
    },
}
with open('results/analysis/confidence_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nFull results saved to results/analysis/confidence_results.json")
