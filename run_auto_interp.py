"""
Part 4: Automated Interpretability via EleutherAI delphi methodology

Pipeline
--------
1.  Load Asian folktale embeddings (computed by run_interpretability.py, or
    re-encode on the fly from the trained classifier).
2.  Train a TopK Sparse Autoencoder (SAE) on the 384-dim embeddings, producing
    a 1536-dim sparse code z for each tale.  The reconstruction is xDz+:
      x̂ = W_dec @ z + b_pre
3.  Build an exemplar bank: for each of the top-N most-active SAE features,
    collect the k highest-activation tales and k zero-activation tales.
4.  Call Claude (Anthropic API) on each feature:
      a.  Generate a one-sentence description of what the feature detects.
      b.  Score the description: predict activation on held-out tales.
5.  Also run cluster-level auto-interp on the existing k-means clusters
    (no SAE required; treats cluster membership as a binary activation).
6.  Save all results to results/analysis/auto_interp_results.json

Usage
-----
    # Full run (SAE + LLM explanations)
    ANTHROPIC_API_KEY=sk-... python run_auto_interp.py

    # Skip LLM explanations (train SAE only, generate sparse features)
    python run_auto_interp.py --no-llm

    # Explain only existing k-means clusters (skip SAE training)
    ANTHROPIC_API_KEY=sk-... python run_auto_interp.py --clusters-only

    # Re-use a previously saved SAE checkpoint
    ANTHROPIC_API_KEY=sk-... python run_auto_interp.py --load-sae results/models/sae.pt

Environment
-----------
    ANTHROPIC_API_KEY   required unless --no-llm is passed
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import FolktaleDataLoader
from src.model import FolktaleClassifier
from src.sae import train_sae, load_sae, get_sparse_activations
from src.auto_interp import (
    AutoInterp,
    FeatureExemplarBank,
    explain_clusters,
    _save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Run auto-interp on folktale embeddings")
    p.add_argument("--no-llm", action="store_true",
                   help="Train SAE and save sparse features but skip LLM calls")
    p.add_argument("--clusters-only", action="store_true",
                   help="Explain k-means clusters only (skip SAE training)")
    p.add_argument("--load-sae", metavar="PATH", default=None,
                   help="Load a pre-trained SAE checkpoint instead of training a new one")
    p.add_argument("--n-features-to-explain", type=int, default=50,
                   help="Number of top SAE features to explain (default: 50)")
    p.add_argument("--sae-n-features", type=int, default=1536,
                   help="SAE hidden dimension (default: 1536 = 4 × 384)")
    p.add_argument("--sae-k", type=int, default=16,
                   help="SAE TopK sparsity (default: 16)")
    p.add_argument("--sae-epochs", type=int, default=150,
                   help="SAE training epochs (default: 150)")
    p.add_argument("--api-key", default=None,
                   help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model for explanations (default: claude-haiku-4-5-20251001)")
    p.add_argument("--skip-scoring", action="store_true",
                   help="Skip the explanation-scoring step (faster)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_or_encode_embeddings(config, output_dir: Path):
    """Load cached embeddings or encode Asian tales from scratch."""
    emb_path = output_dir / "asian_embeddings.npy"
    if emb_path.exists():
        logger.info(f"Loading cached embeddings from {emb_path}")
        embeddings = np.load(str(emb_path))
        # Also load the corresponding texts
        loader = FolktaleDataLoader("config/config.yaml")
        asian_df = loader.load_asian_tales()
        texts = asian_df["text"].values.tolist()
        return embeddings, texts, asian_df

    logger.info("Embeddings not found — encoding from scratch")
    loader = FolktaleDataLoader("config/config.yaml")
    asian_df = loader.load_asian_tales()
    texts = asian_df["text"].values.tolist()

    clf_path = "results/models/folktale_classifier.pkl"
    if not Path(clf_path).exists():
        raise FileNotFoundError(
            f"Trained classifier not found at {clf_path}. "
            "Run run_evaluation.py first."
        )
    clf = FolktaleClassifier(config, device="cpu")
    clf.load(clf_path)
    embeddings = clf.encode_texts(texts)
    np.save(str(emb_path), embeddings)
    logger.info(f"Embeddings saved → {emb_path}")
    return embeddings, texts, asian_df


def _select_top_features(activations: np.ndarray, n: int) -> list:
    """Select the n most active (non-dead) features by number of tales activated."""
    n_active_per_feature = (activations > 0).sum(axis=0)
    order = np.argsort(n_active_per_feature)[::-1]
    # Exclude fully dead features
    order = [f for f in order if n_active_per_feature[f] > 0]
    return list(order[:n])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = Path("results/analysis")
    models_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PART 4: AUTO-INTERPRETABILITY (EleutherAI delphi methodology)")
    print("=" * 80)

    # ── Step 1: Load embeddings ───────────────────────────────────────────────
    print("\n[1/5] Loading Asian folktale embeddings …")
    embeddings, texts, asian_df = _load_or_encode_embeddings(config, output_dir)
    n_tales, d_model = embeddings.shape
    print(f"      {n_tales} tales  ×  {d_model} dims")

    # ── Step 2: Cluster-level auto-interp (always run) ────────────────────────
    cluster_assignments = None
    cluster_explanations = {}
    ca_path = output_dir / "cluster_assignments.npy"
    if ca_path.exists():
        cluster_assignments = np.load(str(ca_path))
        print(f"\n[2/5] Loaded {int(cluster_assignments.max())+1} k-means clusters")

        if not args.no_llm:
            api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                print("      ⚠  ANTHROPIC_API_KEY not set — skipping cluster explanations")
            else:
                print("      Generating cluster explanations via LLM …")
                cluster_out = str(output_dir / "cluster_explanations.json")
                cluster_explanations = explain_clusters(
                    cluster_assignments=cluster_assignments,
                    texts=texts,
                    metadata=asian_df,
                    api_key=api_key,
                    model=args.model,
                    output_path=cluster_out,
                )
                print(f"      Saved → {cluster_out}")
                for cid, exp in sorted(cluster_explanations.items()):
                    n = int((cluster_assignments == cid).sum())
                    print(f"        Cluster {cid:2d} (n={n:4d}): {exp}")
    else:
        print("\n[2/5] No cluster_assignments.npy found — run run_interpretability.py first")

    if args.clusters_only:
        print("\n--clusters-only flag set. Done.")
        return

    # ── Step 3: Train / load SAE ─────────────────────────────────────────────
    sae_path = args.load_sae or str(models_dir / "sae.pt")

    if args.load_sae and Path(args.load_sae).exists():
        print(f"\n[3/5] Loading SAE from {args.load_sae} …")
        sae = load_sae(args.load_sae, device="cpu")
    elif Path(sae_path).exists() and not args.load_sae:
        print(f"\n[3/5] Loading cached SAE from {sae_path} …")
        sae = load_sae(sae_path, device="cpu")
    else:
        print(f"\n[3/5] Training TopK SAE "
              f"({d_model} → {args.sae_n_features} features, k={args.sae_k}, "
              f"epochs={args.sae_epochs}) …")
        sae, history = train_sae(
            embeddings=embeddings,
            n_features=args.sae_n_features,
            k=args.sae_k,
            n_epochs=args.sae_epochs,
            lr=2e-4,
            device="cpu",
            checkpoint_path=sae_path,
            seed=config["model"]["random_state"],
        )
        print(f"      Variance explained: {history['var_explained']:.4f}")
        print(f"      Dead features at final epoch: {history['dead_features'][-1]}/{args.sae_n_features}")
        # Save training history
        with open(output_dir / "sae_training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Step 4: Get sparse activations ───────────────────────────────────────
    print("\n[4/5] Computing sparse SAE activations …")
    sparse_acts = get_sparse_activations(sae, embeddings, device="cpu")
    np.save(str(output_dir / "sae_activations.npy"), sparse_acts)
    n_active_features = int((sparse_acts.max(axis=0) > 0).sum())
    print(f"      Activations shape: {sparse_acts.shape}")
    print(f"      Active features (≥1 tale): {n_active_features}/{args.sae_n_features}")
    print(f"      Sparsity: {(sparse_acts > 0).mean():.4f} non-zero entries")

    # ── Step 5: Auto-interp via LLM ───────────────────────────────────────────
    sae_results: Dict = {}
    if args.no_llm:
        print("\n[5/5] --no-llm set. Skipping LLM explanation step.")
        print(f"      Sparse activations saved → {output_dir}/sae_activations.npy")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\n[5/5] ANTHROPIC_API_KEY not set. Skipping LLM step.")
            print("      Set the key and re-run to generate explanations.")
        else:
            top_features = _select_top_features(sparse_acts, args.n_features_to_explain)
            print(f"\n[5/5] Explaining top {len(top_features)} SAE features via {args.model} …")

            bank = FeatureExemplarBank(
                activations=sparse_acts,
                texts=texts,
                metadata=asian_df,
                n_pos=7,
                n_neg=5,
            )
            interp = AutoInterp(
                api_key=api_key,
                model=args.model,
                rate_limit_delay=0.5,
            )
            sae_out_path = str(output_dir / "auto_interp_results.json")
            sae_results = interp.run(
                bank,
                feature_ids=top_features,
                skip_scoring=args.skip_scoring,
                output_path=sae_out_path,
            )

            # Print summary
            print(f"\n  {'Feature':>8}  {'Active tales':>13}  {'Score':>6}  Explanation")
            print("  " + "-" * 78)
            for fid in top_features:
                entry = sae_results[fid]
                n_act = entry["stats"]["n_active"]
                score = entry.get("score")
                score_str = f"{score:.2f}" if score is not None else "  — "
                exp = (entry["explanation"] or "")[:70]
                print(f"  {fid:>8}  {n_act:>13}  {score_str:>6}  {exp}")

    # ── Save combined results ─────────────────────────────────────────────────
    combined = {
        "metadata": {
            "n_tales": n_tales,
            "embedding_dim": d_model,
            "sae_n_features": args.sae_n_features,
            "sae_k": args.sae_k,
            "n_features_explained": len(sae_results),
            "model": args.model,
        },
        "cluster_explanations": {str(k): v for k, v in cluster_explanations.items()},
        "sae_feature_explanations": {
            str(fid): {
                "explanation": e["explanation"],
                "stats": e["stats"],
                "score": e.get("score"),
                "accuracy": e.get("accuracy"),
            }
            for fid, e in sae_results.items()
        },
    }
    out_path = output_dir / "auto_interp_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Embeddings:            {n_tales} tales × {d_model} dims")
    print(f"  SAE:                   {d_model} → {args.sae_n_features} features  (k={args.sae_k})")
    print(f"  Active SAE features:   {n_active_features}")
    print(f"  Features explained:    {len(sae_results)}")
    print(f"  Cluster explanations:  {len(cluster_explanations)}")
    print(f"\n  Results saved → {out_path}")

    if sae_results:
        scored = [e["score"] for e in sae_results.values() if e.get("score") is not None]
        if scored:
            print(f"\n  Explanation quality (F1 on held-out tales):")
            print(f"    mean={np.mean(scored):.3f}  median={np.median(scored):.3f}  "
                  f"min={np.min(scored):.3f}  max={np.max(scored):.3f}")


if __name__ == "__main__":
    main()
