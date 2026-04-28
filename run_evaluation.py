"""
Rigorous evaluation: ATU classifier trained on Western tales, tested on Asian tales.

Two-level methodology:
  1. In-distribution gate: single 70/10/20 split, prove the model learned ATU
     structure (macro F1 > 2× random) and measure calibration (ECE).
  2. Cross-validated confidence comparison: 5-fold CV on all Western tales,
     pool held-out confidence scores (each tale held out exactly once), compare
     to Asian tale confidence with Mann-Whitney U test.

Run from the project root: py run_evaluation.py
"""

import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import yaml
import json
import logging
from pathlib import Path

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.bool_):      return bool(obj)
        return super().default(obj)
from sklearn.model_selection import train_test_split
from scipy import stats

from src.data_loader import FolktaleDataLoader
from src.model import FolktaleClassifier, atu_scholarly_category, ATU_LABELS

logging.basicConfig(level=logging.WARNING)

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# ── Western tales ─────────────────────────────────────────────────────────────
print("Loading Western folktales …")
western_df = pd.read_csv('../aft.csv')
western_df['text_processed'] = western_df['text'].fillna('').apply(
    lambda t: ' '.join(t.split()))
western_df['atu_label'] = western_df['atu_id'].apply(atu_scholarly_category)
western_df = western_df[western_df['atu_label'].notna()].reset_index(drop=True)

texts_all  = western_df['text_processed'].values
labels_all = western_df['atu_label'].values

class_dist = western_df['atu_label'].value_counts()
K_true = class_dist.shape[0]
print(f"  {len(texts_all)} tales → {K_true} ATU categories  "
      f"(random baseline = {1/K_true:.4f})")
for cat, n in class_dist.items():
    print(f"    {ATU_LABELS.get(cat, cat):<58}  {n:>4} tales")

# ── Level 1: In-distribution gate (single 70/10/20 split) ────────────────────
# We need an independent held-out set to prove the model learned ATU structure
# before we run the CV confidence comparison. Using a fixed 20% test set here
# avoids any label-leakage between the gate and the CV confidence step.
print("\n── Level 1: In-distribution gate (held-out 20% test set) ──────────────────")
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts_all, labels_all, test_size=0.2, random_state=42, stratify=labels_all)
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train, labels_train, test_size=0.125, random_state=42, stratify=labels_train)
# 0.125 × 0.8 = 0.10 → 70/10/20 split

print("Training classifier on 70% …")
clf = FolktaleClassifier(config, device='cpu')
metrics = clf.train(texts_train.tolist(), labels_train.tolist(),
                    texts_val.tolist(), labels_val.tolist())
print(f"  Train acc={metrics['train_accuracy']:.3f}  "
      f"Val acc={metrics.get('val_accuracy', float('nan')):.3f}  "
      f"n_classes={metrics['n_classes']}")

print("\nIn-distribution evaluation on Western held-out test set …")
indist = clf.evaluate_indistribution(texts_test.tolist(), labels_test.tolist())

print(f"  n_classes     : {indist['n_classes']}   (random baseline = {indist['random_baseline']:.4f})")
print(f"  Accuracy      : {indist['accuracy']:.4f}")
print(f"  Macro F1      : {indist['macro_f1']:.4f}   ← uniform weight across all ATU classes")
print(f"  Weighted F1   : {indist['weighted_f1']:.4f}   ← weighted by class frequency")
print(f"  ECE           : {indist['ece']:.4f}   ← calibration error; 0.00 = perfect")

report   = indist['classification_report']
per_class = {k: v for k, v in report.items()
             if k not in ('accuracy', 'macro avg', 'weighted avg')}
worst = sorted(per_class.items(), key=lambda x: x[1]['f1-score'])[:10]
print("\n  10 worst ATU classes by F1:")
print(f"  {'ATU class':<20} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}")
for cls, vals in worst:
    print(f"  {str(cls):<20} {vals['precision']:>10.3f} {vals['recall']:>8.3f} "
          f"{vals['f1-score']:>8.3f} {int(vals['support']):>9}")

MACRO_F1_GATE = 2.0 / indist['n_classes']
if indist['macro_f1'] < MACRO_F1_GATE:
    print(f"\n  !! Macro F1 {indist['macro_f1']:.4f} < gate {MACRO_F1_GATE:.4f} (2× random).")
    print("     Classifier has not learned ATU structure — OOD comparison would be uninterpretable.")
    sys.exit(1)
print(f"\n  ✓ Macro F1 {indist['macro_f1']:.4f} >= gate {MACRO_F1_GATE:.4f}. Gate passed.")

# ── Level 2: Cross-validated Western confidence baseline ─────────────────────
# Each Western tale is held out exactly once; its confidence score comes from a
# model that never trained on it. This is the rigorous unbiased baseline.
print("\n── Level 2: Cross-validated confidence baseline (5-fold, ALL Western tales) ──")
cv_result = clf.cross_validated_confidence(
    texts_all.tolist(), labels_all.tolist(), n_splits=5)

print(f"\n  CV accuracies per fold: {[f'{a:.4f}' for a in cv_result['cv_accuracies']]}")
print(f"  Mean CV accuracy       : {cv_result['mean_cv_accuracy']:.4f}  ← paper accuracy claim")
print(f"  CV macro F1 per fold   : {[f'{f:.4f}' for f in cv_result['cv_macro_f1s']]}")
print(f"  Mean CV macro F1       : {cv_result['mean_cv_macro_f1']:.4f}")
print(f"  Western tales used     : {len(texts_all)}  (all, no leakage)")

probs_west_cv = cv_result['probs']
mh_west_cv    = cv_result['mh_distances']
m_west_cv     = clf.get_confidence_metrics(probs_west_cv)

K = probs_west_cv.shape[1]
print(f"\n  n_classes K={K}   max possible entropy = log({K}) = {np.log(K):.3f} nats")

# ── Asian tales ───────────────────────────────────────────────────────────────
print("\nLoading Asian/SE Asian folktales …")
asian_records = []
for json_path in [
    '../china_china_fables_dataset (1).json',
    '../japan_japan_fairy_tales_dataset.json',
    '../korea_korea_fairy_tales_dataset.json',
]:
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    region = Path(json_path).stem.split('_')[1]
    for rec in data:
        body = rec.get('body') or rec.get('text') or ''
        if body.strip():
            asian_records.append({'region': region, 'text': body.strip(),
                                  'title': rec.get('title', '')})

# ── Additional Asian sources (new batch) ──────────────────────────────────────
_ASIAN_EXTRA_DIR = Path('../asian (2)/asian')

# mftd_chinese.json uses 'story' field instead of 'text'/'body'
with open(_ASIAN_EXTRA_DIR / 'mftd_chinese.json', encoding='utf-8') as f:
    for rec in json.load(f):
        body = rec.get('story') or ''
        if body.strip():
            asian_records.append({'region': rec.get('region', 'china').lower(),
                                  'text': body.strip(),
                                  'title': rec.get('title', '')})

# CSV sources: all share columns title, text, region (plus optional source)
_METADATA_TITLES = {'KOREAN FOLK TALES', 'PREFACE', 'CONTENTS', 'BIOGRAPHICAL'}
for _csv in [
    'philippine_cole.csv',
    'chinese_macgowan.csv',
    'chinese_wilhelm.csv',
    'gutenberg_japanese.csv',
    'japanese_nixon.csv',
    'japanese_ozaki.csv',
    'korean_griffis.csv',
    'korean_im_bang.csv',
    'tibetan_oconnor.csv',
    'mongolian_busk.csv',
    'thai_laos_fleeson.csv',
    'myanmar_shan_griggs.csv',
    'malay_skeat.csv',
]:
    _df = pd.read_csv(_ASIAN_EXTRA_DIR / _csv)
    for _, row in _df.iterrows():
        title = str(row.get('title', ''))
        if title in _METADATA_TITLES:
            continue
        body = str(row.get('text', '') or '').strip()
        if body:
            asian_records.append({'region': str(row['region']).lower(),
                                  'text': body,
                                  'title': title})

asian_df = pd.DataFrame(asian_records)
# Remove Tibet (excluded from regional analysis)
asian_df = asian_df[asian_df['region'] != 'tibet'].reset_index(drop=True)
print(f"  {len(asian_df)} tales  ({asian_df['region'].value_counts().to_dict()})")

# Asian confidence from the final model (trained on all Western tales)
print("\nFitting final model on all Western tales for Asian inference …")
clf_final = FolktaleClassifier(config, device='cpu')
clf_final.train(texts_all.tolist(), labels_all.tolist())

texts_asian = asian_df['text'].values.tolist()
_, probs_asian = clf_final.predict(texts_asian)
m_asian  = clf_final.get_confidence_metrics(probs_asian)
mh_asian = clf_final.mahalanobis_distances(texts_asian)

# ── Per-region breakdown ──────────────────────────────────────────────────────
print("\nPer-region normalized entropy (Asian tales):")
for region, grp in asian_df.groupby('region'):
    idx = grp.index.tolist()
    ne  = m_asian['normalized_entropy'][idx].mean()
    print(f"  {region:8s}: mean normalized entropy = {ne:.4f}")

# ── Comparison table ──────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled

# Mahalanobis: re-compute Western baseline using the cv model's distances
# (we already have mh_west_cv from the CV step)
print("\n" + "="*95)
print("  Hypothesis test: is the ATU classifier less confident on Asian tales?")
print("  Western baseline = cross-validated held-out scores (n={})".format(len(texts_all)))
print("="*95)

rows = [
    ("Max-softmax confidence",    m_west_cv['max_confidence'],      m_asian['max_confidence'],      "higher=more confident"),
    ("Normalized entropy H/logK", m_west_cv['normalized_entropy'],  m_asian['normalized_entropy'],  "lower=more confident; 1=random"),
    ("Margin p1-p2",               m_west_cv['margin'],              m_asian['margin'],              "higher=more decisive"),
    ("Mahalanobis distance",       mh_west_cv,                       mh_asian,                       "higher=more OOD"),
]
print(f"\n{'Metric':<34} {'Western':>9} {'Asian':>9} {'Delta':>9} {'Cohen d':>9}  {'MW p':>10}  Note")
print("-"*95)
for label, w, a, note in rows:
    delta = np.mean(w) - np.mean(a)
    d     = cohens_d(w, a)
    _, p  = stats.mannwhitneyu(w, a, alternative='two-sided')
    print(f"{label:<34} {np.mean(w):>9.4f} {np.mean(a):>9.4f} {delta:>+9.4f} {d:>9.3f}  {p:>10.2e}  {note}")

# ── Interpretation ─────────────────────────────────────────────────────────────
ne_w = m_west_cv['mean_normalized_entropy']
ne_a = m_asian['mean_normalized_entropy']
_, p_ne = stats.mannwhitneyu(m_west_cv['normalized_entropy'],
                              m_asian['normalized_entropy'], alternative='two-sided')
d_ne = cohens_d(m_west_cv['normalized_entropy'], m_asian['normalized_entropy'])

print(f"\n── Interpretation ──────────────────────────────────────────────────────────────────────")
print(f"  Western CV accuracy (paper claim) : {cv_result['mean_cv_accuracy']:.4f}  "
      f"({cv_result['mean_cv_accuracy']*100:.1f}%)  mean over 5 held-out folds")
print(f"  Western normalized entropy        : {ne_w:.4f}  (cross-validated baseline)")
print(f"  Asian   normalized entropy        : {ne_a:.4f}  (model trained on all Western tales)")
print(f"  Scale: 0.00 = perfectly certain  |  1.00 = pure random guess over {K} classes")

thresholds = [
    (0.85, "near-random (ATU does not generalize at all)"),
    (0.65, "highly uncertain (ATU generalizes poorly)"),
    (0.40, "moderately uncertain (partial generalization)"),
    (0.00, "relatively confident (ATU may generalize)"),
]
for cutoff, label in thresholds:
    if ne_a > cutoff:
        print(f"  Asian H_norm={ne_a:.3f} => {label}")
        break

effect = ("large" if abs(d_ne) >= 0.8 else
          "medium" if abs(d_ne) >= 0.5 else
          "small"  if abs(d_ne) >= 0.2 else "negligible")
print(f"  Gap Western→Asian: Δ={ne_w-ne_a:+.4f}, Cohen's d={d_ne:.3f} ({effect} effect)")
print(f"  Mann-Whitney p={p_ne:.2e}  → hypothesis {'SUPPORTED' if p_ne < 0.05 and ne_a > ne_w else 'NOT supported'} at α=0.05")

# ── Save results ───────────────────────────────────────────────────────────────
Path('results/analysis').mkdir(parents=True, exist_ok=True)
out = {
    'methodology': {
        'grouping':       '14_scholarly_atu',
        'grouping_note':  (
            'Raw ATU IDs mapped to 14 scholarly subdivisions following Uther (2004). '
            'Raw IDs (160 classes, ~8 examples each) gave ~25% CV accuracy; '
            '7-class grouping was too coarse for narrative analysis. '
            '14-class achieves ~70% CV accuracy on 1484 examples.'
        ),
        'western_baseline': (
            '5-fold cross-validated: each Western tale held out exactly once. '
            'No confidence score is ever measured on a tale the model was trained on.'
        ),
        'asian_inference': (
            'Final model trained on all 1484 Western tales. '
            'Asian tales have no overlap with training data by construction.'
        ),
    },
    'n_atu_classes': K,
    'max_entropy':   float(np.log(K)),
    'cross_validation': {
        'n_splits':         5,
        'cv_accuracies':    cv_result['cv_accuracies'],
        'cv_macro_f1s':     cv_result['cv_macro_f1s'],
        'mean_cv_accuracy': cv_result['mean_cv_accuracy'],
        'mean_cv_macro_f1': cv_result['mean_cv_macro_f1'],
        'n_western_tales':  len(texts_all),
    },
    'indistribution_eval': {
        'accuracy':        indist['accuracy'],
        'macro_f1':        indist['macro_f1'],
        'weighted_f1':     indist['weighted_f1'],
        'ece':             indist['ece'],
        'random_baseline': indist['random_baseline'],
        'n_samples':       indist['n_samples'],
        'n_classes':       indist['n_classes'],
        'note': '20% held-out test set, independent of CV confidence step',
    },
    'western_cv': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in m_west_cv.items()},
    'asian': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in m_asian.items()},
    'mahalanobis': {
        'western_cv': mh_west_cv.tolist(),
        'asian':      mh_asian.tolist(),
        'note': 'Western = CV held-out per-fold; Asian = final model trained on all Western',
    },
    'per_region_normalized_entropy': {
        region: float(m_asian['normalized_entropy'][grp.index].mean())
        for region, grp in asian_df.groupby('region')
    },
    'hypothesis_test': {
        'metric':          'normalized_entropy',
        'western_mean':    ne_w,
        'asian_mean':      ne_a,
        'delta':           ne_w - ne_a,
        'cohens_d':        d_ne,
        'mannwhitney_p':   p_ne,
        'effect_size':     effect,
        'supported':       bool(p_ne < 0.05 and ne_a > ne_w),
    },
}
with open('results/analysis/confidence_results.json', 'w') as f:
    json.dump(out, f, indent=2, cls=_NumpyEncoder)
print("\nFull results saved to results/analysis/confidence_results.json")
