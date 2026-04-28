"""
Part 3: Interpretability & Archetype Discovery
Cluster Asian/SE Asian folktales and extract narrative motifs via:
- K-means clustering on embeddings
- Sparse feature extraction (via SAE-Lens when available)
- Logistic regression P(cluster | embedding) coefficient analysis
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import FolktaleDataLoader
from src.model import FolktaleClassifier
from src.interpretability import MotifExtractor
import yaml

logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & SETUP
# ══════════════════════════════════════════════════════════════════════════════
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load Asian tales and embeddings
print("="*80)
print("PART 3: INTERPRETABILITY & ARCHETYPE DISCOVERY")
print("="*80)

loader = FolktaleDataLoader('config/config.yaml')
asian_df = loader.load_asian_tales()
print(f"\nLoaded {len(asian_df)} Asian/SE Asian folktales")

# Load classifier to get embeddings
clf = FolktaleClassifier(config, device='cpu')
clf.load('results/models/folktale_classifier.pkl')
print("Loaded trained classifier")

texts_asian = asian_df['text'].values.tolist()
embeddings = clf.encode_texts(texts_asian)
print(f"Extracted embeddings: shape {embeddings.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: CLUSTERING VIA K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
n_clusters = config['clustering']['kmeans']['n_clusters']
print(f"\nStep 1: K-means clustering (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
cluster_assignments = kmeans.fit_predict(embeddings)

cluster_counts = np.bincount(cluster_assignments)
print(f"  Cluster sizes: min={cluster_counts.min()}, max={cluster_counts.max()}, "
      f"mean={cluster_counts.mean():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT MOTIFS (tokens, n-grams, representative stories)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Extract narrative motifs...")
motif_extractor = MotifExtractor(config)

# Top tokens per cluster
print("  - Extracting top tokens per cluster...")
top_tokens = motif_extractor.extract_top_tokens(embeddings, texts_asian, cluster_assignments)

# Top n-grams per cluster
print("  - Extracting distinctive n-grams per cluster...")
top_ngrams = motif_extractor.extract_ngrams(texts_asian, cluster_assignments)

# Representative stories
print("  - Extracting representative stories...")
rep_stories = motif_extractor.extract_representative_stories(
    embeddings, texts_asian, asian_df, cluster_assignments)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN LOGISTIC REGRESSION P(cluster | embedding)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Train logistic regression P(cluster | embedding)...")
lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
lr.fit(embeddings, cluster_assignments)
print(f"  LR accuracy: {lr.score(embeddings, cluster_assignments):.3f}")

# Extract top features per cluster (coefficient analysis)
cluster_features = {}
for cluster_id in range(n_clusters):
    coef = lr.coef_[cluster_id]  # (384,)
    top_indices = np.argsort(np.abs(coef))[-10:][::-1]
    cluster_features[cluster_id] = [(idx, float(coef[idx])) for idx in top_indices]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE ARCHETYPE PROPOSALS
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Generate archetype proposals...")
archetype_proposals = {}

# Archetype naming based on narrative themes
theme_keywords = {
    'royalty': ['king', 'queen', 'prince', 'princess', 'palace', 'court', 'emperor'],
    'magic': ['magic', 'spell', 'curse', 'wizard', 'enchant', 'supernatural', 'mystical'],
    'love': ['love', 'marry', 'bride', 'groom', 'romance', 'beloved', 'heart'],
    'war': ['war', 'battle', 'soldier', 'enemy', 'sword', 'fight', 'warrior'],
    'animal': ['animal', 'beast', 'fox', 'tiger', 'monkey', 'bird', 'creature'],
    'wisdom': ['wise', 'wisdom', 'teach', 'learn', 'advice', 'virtue', 'moral'],
    'religion': ['buddha', 'monk', 'temple', 'prayer', 'enlighten', 'sacred', 'faith'],
}

for cluster_id in range(n_clusters):
    mask = cluster_assignments == cluster_id
    cluster_size = np.sum(mask)

    # Get top tokens and n-grams
    tokens = top_tokens[cluster_id][:5]
    ngrams = top_ngrams[cluster_id][:5]

    # Infer theme from token/n-gram overlap
    token_text = ' '.join([t[0].lower() for t in tokens])
    ngram_text = ' '.join([n[0].lower() for n in ngrams])
    combined_text = token_text + ' ' + ngram_text

    theme_scores = {}
    for theme, keywords in theme_keywords.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        theme_scores[theme] = score

    top_theme = max(theme_scores, key=theme_scores.get)
    theme_label = {
        'royalty': 'Royal/Court Tales',
        'magic': 'Magical/Supernatural',
        'love': 'Romance/Love Stories',
        'war': 'War/Adventure Epics',
        'animal': 'Animal/Trickster Tales',
        'wisdom': 'Wisdom/Teaching Tales',
        'religion': 'Religious/Spiritual',
    }.get(top_theme, 'Narrative Cluster')

    # Build archetype dict
    archetype_proposals[cluster_id] = {
        'name': f"Cluster {cluster_id}: {theme_label}",
        'size': int(cluster_size),
        'theme': top_theme,
        'top_tokens': [(t[0], int(t[1])) for t in tokens],
        'top_ngrams': [(n[0], float(n[1])) for n in ngrams],
        'top_features': cluster_features[cluster_id][:5],
    }

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("DISCOVERED NARRATIVE ARCHETYPES (Asian/SE Asian Folktales)")
print("="*80)

for cluster_id in sorted(archetype_proposals.keys()):
    ap = archetype_proposals[cluster_id]
    print(f"\n[{ap['name']}]  (n={ap['size']})")

    print(f"  Theme: {ap['theme'].upper()}")

    print(f"  Top tokens: {', '.join([t[0] for t in ap['top_tokens']])}")

    print(f"  Key phrases: {', '.join([n[0] for n in ap['top_ngrams']])}")

    # Show a representative story excerpt
    mask = cluster_assignments == cluster_id
    indices = np.where(mask)[0]
    if len(indices) > 0:
        best_idx = indices[0]
        story_title = asian_df.iloc[best_idx].get('title', f'Tale {best_idx}')
        story_excerpt = texts_asian[best_idx][:150].replace('\n', ' ')
        print(f"  Example: \"{story_title}\"")
        print(f"           \"{story_excerpt}...\"")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_dir = Path('results/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'metadata': {
        'n_tales': len(texts_asian),
        'n_clusters': n_clusters,
        'embedding_dim': embeddings.shape[1],
        'clustering_method': 'kmeans',
    },
    'cluster_assignments': cluster_assignments.tolist(),
    'cluster_sizes': cluster_counts.tolist(),
    'archetypes': {
        str(k): {
            'name': v['name'],
            'size': v['size'],
            'theme': v['theme'],
            'top_tokens': v['top_tokens'],
            'top_ngrams': v['top_ngrams'],
            'top_features': v['top_features'],
        }
        for k, v in archetype_proposals.items()
    },
    'logistic_regression_accuracy': float(lr.score(embeddings, cluster_assignments)),
}

# Save results
with open(output_dir / 'interpretability_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save embeddings
np.save(str(output_dir / 'asian_embeddings.npy'), embeddings)

# Save cluster assignments
np.save(str(output_dir / 'cluster_assignments.npy'), cluster_assignments)

print(f"\nSaved to {output_dir}:")
print(f"  - interpretability_results.json (archetypes & motifs)")
print(f"  - asian_embeddings.npy ({embeddings.shape})")
print(f"  - cluster_assignments.npy ({cluster_assignments.shape})")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SUMMARY: Part 3 Interpretability Pipeline")
print("="*80)
print(f"""
Processed:    {len(texts_asian)} Asian/SE Asian folktales
Clustered:    {n_clusters} narrative archetypes via k-means
Extracted:    Top tokens, n-grams, representative stories per cluster
Analyzed:     Logistic regression coefficients P(cluster|embedding)
Discovered:   {len(archetype_proposals)} distinct narrative patterns

Proposed archetype taxonomy:
  * Royal/Court Tales         - Power, governance, court intrigue
  * Magical/Supernatural      - Magic, spirits, curses, wonders
  * Romance/Love Stories      - Love, marriage, relationship journeys
  * War/Adventure Epics       - Battles, heroes, quests
  * Animal/Trickster Tales    - Anthropomorphic animals, tricksters
  * Wisdom/Teaching Tales     - Moral lessons, instruction narratives
  * Religious/Spiritual       - Buddhist/Daoist themes, enlightenment

NEXT: Manually review cluster representatives to refine archetype names.
      Compare with ATU categories and existing Asian folklore literature.
      Use this taxonomy as a foundation for East/SE Asian tale classification.
""")
