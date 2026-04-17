# Quick Start Guide

## 1. Environment Setup

### Create Virtual Environment
```bash
cd folktale-classifier
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; import transformers; import sbert; print('✓ All packages installed')"
```

Or run the validation script:
```bash
python setup_validation.py
```

---

## 2. Data Collection

### Western Folktales (ATU-labeled)
1. Download the AFT dataset from:
   ```
   https://github.com/j-hagedorn/trilogy/blob/master/data/aft.csv
   ```
2. Place in: `data/western/aft.csv`

### Asian/Southeast Asian Folktales
Your Asian folktales are already available as JSON files in the Downloads folder:
- `china_china_fables_dataset (1).json`
- `korea_korea_fables_dataset.json`
- `japan_japan_fables_dataset.json`

**JSON Structure Expected:**
```json
[
  {"text": "Once upon a time in ancient China...", "title": "The Magic Fox"},
  {"text": "In a faraway kingdom...", "title": "The Wise Old Man"}
]
```

Each JSON file should contain an array of tale objects with at least a `text` field.

---

## 3. Running the Pipeline

The project is organized into 3 phases:

### Phase 1: Exploratory Data Analysis (EDA)
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```
**What it does:**
- Loads Western folktales with ATU labels
- Computes text statistics (length, word count, etc.)
- Visualizes ATU category distribution
- Loads and analyzes Asian/SE Asian folktales
- Checks data quality

### Phase 2: Classifier Training & Confidence Testing
```bash
jupyter notebook notebooks/02_classifier_training.ipynb
```
**What it does:**
- Trains logistic regression classifier on Western folktales
- Encodes texts using sentence transformers (384-dim embeddings)
- Tests classifier on Asian/SE Asian folktales
- Measures confidence (softmax max probability)
- Computes Shannon entropy
- **Tests hypothesis**: Does ATU generalize to Asian folktales?

### Phase 3: Neural Network Interpretability & Motif Extraction
```bash
jupyter notebook notebooks/03_interpretability_clustering.ipynb
```
**What it does:**
- Extracts embeddings from hidden layer of sentence transformer
- Clusters Asian folktales using k-means (SAE-lens when available)
- Extracts top-activating tokens per cluster (motif indicators)
- Analyzes distinctive n-grams per cluster
- Identifies representative stories per cluster
- Proposes narrative archetypes for Asian/SE Asian tales

---

## 4. Configuration

Edit `config/config.yaml` to adjust:
- Sentence transformer model
- Logistic regression hyperparameters
- Clustering settings (n_clusters, method)
- Interpretability parameters (top-k tokens, n-grams, etc.)

---

## 5. Understanding the Output

### Directory Structure After Running
```
results/
├── models/
│   └── folktale_classifier.pkl          # Trained classifier
├── embeddings/
│   └── asian_folktales.npy              # Extracted embeddings
├── clusters/
│   └── cluster_assignments.npy           # Cluster labels
├── plots/
│   ├── 01_western_text_stats.png        # Text distributions
│   ├── 02_western_confidence.png        # Confidence on Western tales
│   ├── 02_asian_confidence.png          # Confidence on Asian tales
│   ├── 02_confidence_comparison.png     # Western vs Asian
│   ├── 03_cluster_sizes.png             # Cluster size distribution
│   └── 03_tsne_clusters.png             # 2D t-SNE visualization
└── analysis/
    ├── 02_confidence_analysis.json      # Hypothesis test results
    └── 03_motif_analysis.json           # Extracted motifs & archetypes
```

---

## 6. Key Concepts

### Hypothesis
> A classifier trained on Western folktales with ATU labels will have **low confidence** when applied to Asian/SE Asian folktales, supporting the hypothesis that ATU categories don't generalize well to non-Western narrative structures.

### Motif Extraction
We identify motifs using multiple complementary approaches:
1. **Token Analysis**: Top-activating words that distinguish each cluster
2. **N-gram Analysis**: Distinctive phrases and narrative elements
3. **Representative Stories**: Stories closest to cluster centroids
4. **Logistic Regression Weights**: Features with high discriminative power

### Archetype Discovery
Proposed narrative archetypes emerge from clustering:
- **Royal/Court Tales**: Stories featuring kings, emperors, court intrigue
- **Magical/Supernatural**: Magic, spirits, curses, enchantments
- **Romance/Love Tales**: Love stories, marriages, human relationships
- **War/Adventure**: Battle, heroic quests, warrior tales
- **Animal/Beast Tales**: Anthropomorphic animals, trickster figures
- **Buddhist/Religious**: Religious teachings, enlightenment narratives
- **Confucian Moral**: Moral lessons, filial piety, virtue tales

---

## 7. Troubleshooting

### Issue: CUDA/GPU not available
```python
# In notebooks, use CPU instead:
clf = FolktaleClassifier(config, device='cpu')
```

### Issue: Out of memory during embedding extraction
```python
# Process texts in batches:
batch_size = 32  # Reduce if needed
```

### Issue: Data not loading
- Verify path structure matches `config/config.yaml`
- Check file encoding (should be UTF-8)
- Ensure CSV has expected columns

---

## 8. Further Development

### Enhancements to Consider
1. **Implement SAE-Lens**: Replace k-means with Sparse Autoencoders for better interpretability
2. **Cross-validation**: More rigorous statistical testing of archetype stability
3. **Domain Expert Review**: Have folklorists validate proposed archetypes
4. **Attention Visualization**: Visualize which text spans activate per cluster
5. **Building ATU Alternative**: Design parallel taxonomy for Asian folktales
6. **Multi-language Support**: Extend to original language texts

### Papers to Review
- "Discovering latent narrative structures in storytelling"
- "Neural network interpretability for NLP"
- "Comparative folklore analysis"

---

## 9. Citation

If you use this project, please cite:
```bibtex
@project{folktale_classifier_2024,
  title={Discovering Narrative Archetypes in East/Southeast Asian Folktales via Neural Network Interpretability},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/folktale-classifier}
}
```

---

## Questions?

Refer to the individual notebook docstrings and the [README.md](../README.md) for more details.
