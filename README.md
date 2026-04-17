# East/Southeast Asian Folktale Classifier

## Project Overview

This project investigates whether probabilistic machine learning models and neural networks can discover narrative archetypes in East/Southeast Asian folktales and identify the motifs that represent each cluster. 

### Hypothesis

The Aarne-Thompson-Uther (ATU) Index, designed primarily for European folklore, shows poor generalization to Asian and Southeast Asian tales. We hypothesize that:
1. A classifier trained on Western folktales with ATU labels will have **low confidence** when applied to Asian/Southeast Asian folktales
2. Neural network interpretability techniques (SAE-lens, k-means, Gemmascope) can reveal **distinct narrative archetypes** specific to East/Southwest Asian cultures
3. These discovered patterns can either improve ATU classification or establish a **parallel taxonomy** rooted in Buddhist/Confucian narrative structures

## Project Structure

```
folktale-classifier/
├── data/                    # Raw and processed data
│   ├── western/            # Western folktales with ATU labels
│   ├── asian/              # East/Southeast Asian folktales
│   └── processed/          # Preprocessed text & embeddings
├── src/                     # Core pipeline modules
│   ├── __init__.py
│   ├── data_loader.py      # Data collection & preprocessing
│   ├── model.py            # Logistic regression classifier
│   ├── interpretability.py  # SAE-lens, clustering, motif extraction
│   └── visualization.py     # Results visualization
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_classifier.ipynb  # Model training & testing
│   └── 03_interpretability.ipynb # SAE-lens & clustering
├── config/                  # Configuration files
│   └── config.yaml         # Hyperparameters, paths
├── results/                 # Model outputs, clusters, visualizations
└── requirements.txt         # Python dependencies
```

## Data Sources

1. **Western Folktales**: https://github.com/j-hagedorn/trilogy/blob/master/data/aft.csv
   - 1519 folktales with ATU labels, provenance, and full text
   
2. **Asian/Southeast Asian Folktales**: JSON datasets in parent directory
   - china_china_fables_dataset (1).json
   - korea_korea_fables_dataset.json
   - japan_japan_fables_dataset.json
   - Each JSON contains array of tale objects with 'text' field

## Methodology

### Phase 1: Classifier Training
- Vectorize stories using sentence transformers (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- Train regularized logistic regression: $p(y | X; W)$ where $y$ is ATU category
- Compute softmax confidence for each story

### Phase 2: Confidence Testing
- Apply trained classifier to Asian/SE Asian folktales
- Measure Shannon entropy and max confidence
- Low values support hypothesis that ATU doesn't generalize

### Phase 3: Clustering & Motif Extraction
- Extract embeddings from hidden layer of neural network
- Apply SAE-lens (or k-means/Gemmascope) for sparse clustering
- Identify motifs:
  - Top-activating tokens per SAE feature
  - High-weight input features from logistic regression
  - Representative stories per cluster
  - Contrastive n-gram analysis

## Setup

1. Create a Python 3.9+ virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. Configure paths in `config/config.yaml`

## Running the Pipeline

```bash
# Phase 1: Training
jupyter notebook notebooks/02_classifier.ipynb

# Phase 2: Testing on Asian folktales
python src/model.py --test --input data/asian/

# Phase 3: Clustering & Motif Extraction
jupyter notebook notebooks/03_interpretability.ipynb
```

## Key Papers & References

- Aarne-Thompson-Uther Index: https://en.wikipedia.org/wiki/Aarne%E2%80%93Thompson%E2%80%93Uther_Index
- SAE-Lens: https://github.com/jbloomaus/SAE_Lens
- Sentence Transformers: https://www.sbert.net/
- On narrative archetypes in folklore: See project proposal

## Contributors

[Your name/team]

## License

MIT License
