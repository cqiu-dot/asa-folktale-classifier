"""
Automated interpretability for SAE features, following EleutherAI's delphi methodology.

Pipeline (per feature):
  1.  Collect top-K *positive* exemplars  (highest SAE activation)
      and K *negative* exemplars           (zero / near-zero activation)
  2.  Send both sets to Claude with an explanation-generation prompt
  3.  Score the explanation: Claude is shown held-out examples without
      activation labels and asked to predict which ones should activate
      the feature; score = F1 between its predictions and ground truth
  4.  Results persisted to JSON keyed by feature index

Usage
-----
    from src.auto_interp import AutoInterp, FeatureExemplarBank

    bank = FeatureExemplarBank(activations, texts, metadata)
    interp = AutoInterp(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = interp.run(bank, feature_ids=range(50))

Reference
---------
Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features
in Language Models"; EleutherAI delphi library (github.com/EleutherAI/delphi).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Exemplar bank ──────────────────────────────────────────────────────────────

@dataclass
class Exemplar:
    text: str
    activation: float
    title: str = ""
    region: str = ""
    index: int = -1


class FeatureExemplarBank:
    """
    For each SAE feature, stores the top-K activating and K random negative
    story excerpts along with their activation values.

    Parameters
    ----------
    activations : (n_tales, n_features) sparse activation matrix from the SAE
    texts       : list of story texts (length n_tales)
    metadata    : optional DataFrame with 'title', 'region' columns
    n_pos       : number of high-activation exemplars per feature
    n_neg       : number of zero/near-zero exemplars per feature
    excerpt_len : character length of each excerpt shown to the LLM
    """

    def __init__(
        self,
        activations: np.ndarray,
        texts: List[str],
        metadata=None,
        n_pos: int = 7,
        n_neg: int = 5,
        excerpt_len: int = 400,
        seed: int = 42,
    ):
        self.activations = activations          # (n, F)
        self.texts = texts
        self.metadata = metadata
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.excerpt_len = excerpt_len
        self.rng = np.random.default_rng(seed)

        n_tales, n_features = activations.shape
        logger.info(
            f"FeatureExemplarBank: {n_tales} tales × {n_features} features  "
            f"(pos={n_pos}, neg={n_neg})"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_meta(self, idx: int) -> Tuple[str, str]:
        if self.metadata is None:
            return "", ""
        row = self.metadata.iloc[idx]
        title  = str(row.get("title",  ""))
        region = str(row.get("region", ""))
        return title, region

    def _make_exemplar(self, idx: int, act: float) -> Exemplar:
        title, region = self._get_meta(idx)
        excerpt = self.texts[idx][: self.excerpt_len].replace("\n", " ").strip()
        return Exemplar(
            text=excerpt, activation=act,
            title=title, region=region, index=idx,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def get_exemplars(
        self, feature_id: int
    ) -> Tuple[List[Exemplar], List[Exemplar]]:
        """
        Return (positive_exemplars, negative_exemplars) for one feature.

        Positives: top-n_pos by activation value (must be > 0).
        Negatives: n_neg tales sampled uniformly from those with activation == 0.
        """
        acts = self.activations[:, feature_id]          # (n,)

        # Positive: sort descending by activation, take top-n_pos
        pos_order = np.argsort(acts)[::-1]
        positives = []
        for idx in pos_order:
            if float(acts[idx]) <= 0:
                break
            positives.append(self._make_exemplar(int(idx), float(acts[idx])))
            if len(positives) >= self.n_pos:
                break

        # Negative: zero-activation tales
        zero_idx = np.where(acts == 0)[0]
        chosen = self.rng.choice(
            zero_idx,
            size=min(self.n_neg, len(zero_idx)),
            replace=False,
        )
        negatives = [self._make_exemplar(int(i), 0.0) for i in chosen]

        return positives, negatives

    def feature_stats(self, feature_id: int) -> Dict:
        acts = self.activations[:, feature_id]
        n_active = int((acts > 0).sum())
        return {
            "n_active": n_active,
            "frac_active": float(n_active / len(acts)),
            "mean_active": float(acts[acts > 0].mean()) if n_active else 0.0,
            "max_active": float(acts.max()),
        }

    def dead_features(self) -> List[int]:
        return [f for f in range(self.activations.shape[1])
                if (self.activations[:, f] > 0).sum() == 0]


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXPLAIN_SYSTEM = """\
You are an expert in world folklore and computational narrative analysis.
You will be shown excerpts from folktales that STRONGLY activate a latent
feature in a Sparse Autoencoder (SAE) trained on sentence-transformer
embeddings of Asian and Southeast Asian folktales, alongside excerpts that
do NOT activate this feature.

Your task: Identify the specific narrative element, motif, or thematic
pattern that distinguishes the activating stories from the non-activating ones.
Be precise. Avoid generic descriptions like "all involve characters" —
instead aim for something like:
  "Tales where a trickster animal (fox, monkey, or rabbit) outwits a larger
   predator through clever deception."
"""

_EXPLAIN_USER = """\
Feature {feature_id} — activation statistics: {stats}

══ HIGH-ACTIVATION EXCERPTS (feature fires strongly) ══
{pos_block}

══ NON-ACTIVATING EXCERPTS (feature does not fire) ══
{neg_block}

Describe in ONE sentence what narrative element or theme Feature {feature_id} \
detects, based on the contrast between the two sets above.
"""

_SCORE_SYSTEM = """\
You are evaluating the quality of an automated interpretation of a neural
network feature trained on folktales.  For each test excerpt below, decide
whether it likely activates the feature as described.

Output ONLY a JSON array of 0s and 1s (1 = likely activates, 0 = likely does
not), with one entry per excerpt, in the same order.  Example: [1,0,1,0,1,1,0]
"""

_SCORE_USER = """\
Proposed interpretation of Feature {feature_id}:
"{explanation}"

Test excerpts (in order):
{test_block}

Output the JSON array now.
"""


# ── Auto-interp engine ────────────────────────────────────────────────────────

class AutoInterp:
    """
    EleutherAI delphi-style automated interpretability using the Anthropic API.

    Parameters
    ----------
    api_key         : Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
    model           : Claude model ID to use for explanation generation
    scorer_model    : Claude model ID for scoring (can be a cheaper model)
    explain_retries : number of API retries on transient errors
    rate_limit_delay: seconds to sleep between API calls
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        scorer_model: Optional[str] = None,
        explain_retries: int = 3,
        rate_limit_delay: float = 0.5,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = model
        self.scorer_model = scorer_model or model
        self.explain_retries = explain_retries
        self.rate_limit_delay = rate_limit_delay

    # ── formatting helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_exemplar_block(exemplars: List[Exemplar], show_activation: bool = True) -> str:
        lines = []
        for i, ex in enumerate(exemplars, 1):
            header = f"[{i}]"
            if ex.title:
                header += f" {ex.title}"
            if ex.region:
                header += f" ({ex.region})"
            if show_activation and ex.activation > 0:
                header += f"  act={ex.activation:.3f}"
            lines.append(header)
            lines.append(f'"{ex.text}"')
            lines.append("")
        return "\n".join(lines).strip()

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str, model: str, max_tokens: int = 300) -> str:
        for attempt in range(self.explain_retries):
            try:
                msg = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text.strip()
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt+1}): {exc}. Retrying in {wait}s")
                time.sleep(wait)
        return ""

    def explain_feature(
        self,
        feature_id: int,
        positives: List[Exemplar],
        negatives: List[Exemplar],
        stats: Dict,
    ) -> str:
        """Generate a one-sentence natural-language description of the feature."""
        if not positives:
            return "(dead feature — no activating examples)"

        pos_block = self._format_exemplar_block(positives, show_activation=True)
        neg_block = self._format_exemplar_block(negatives, show_activation=False)
        stats_str = (
            f"active in {stats['n_active']} / {stats.get('n_total', '?')} tales "
            f"({100*stats['frac_active']:.1f}%), "
            f"mean_act={stats['mean_active']:.3f}, max={stats['max_active']:.3f}"
        )

        user_msg = _EXPLAIN_USER.format(
            feature_id=feature_id,
            stats=stats_str,
            pos_block=pos_block,
            neg_block=neg_block,
        )
        time.sleep(self.rate_limit_delay)
        return self._call(_EXPLAIN_SYSTEM, user_msg, self.model, max_tokens=150)

    def score_explanation(
        self,
        feature_id: int,
        explanation: str,
        test_exemplars: List[Exemplar],
        test_labels: List[int],
    ) -> Dict:
        """
        Ask the LLM to predict which test exemplars activate the feature.

        Returns a dict with 'score' (F1), 'predictions', 'labels', 'accuracy'.
        Returns a sentinel dict on failure.
        """
        if not explanation or not test_exemplars:
            return {"score": None, "predictions": [], "labels": test_labels, "accuracy": None}

        test_block_lines = []
        for i, ex in enumerate(test_exemplars, 1):
            title = f" [{ex.title}]" if ex.title else ""
            test_block_lines.append(f"Excerpt {i}{title}: \"{ex.text}\"")
        test_block = "\n\n".join(test_block_lines)

        user_msg = _SCORE_USER.format(
            feature_id=feature_id,
            explanation=explanation,
            test_block=test_block,
        )
        time.sleep(self.rate_limit_delay)
        raw = self._call(_SCORE_SYSTEM, user_msg, self.scorer_model, max_tokens=80)

        # Parse JSON array from response
        try:
            match = re.search(r"\[[\d,\s]+\]", raw)
            if match:
                preds = json.loads(match.group())
            else:
                preds = [int(c) for c in re.findall(r"[01]", raw)]
        except Exception:
            preds = []

        # Pad / truncate to match length
        n = len(test_labels)
        preds = (preds + [0] * n)[:n]

        tp = sum(p == 1 and l == 1 for p, l in zip(preds, test_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, test_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, test_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        acc = sum(p == l for p, l in zip(preds, test_labels)) / n if n else 0.0

        return {
            "score": round(f1, 4),
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "predictions": preds,
            "labels": test_labels,
        }

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def run(
        self,
        bank: FeatureExemplarBank,
        feature_ids: Sequence[int],
        n_score_pos: int = 5,
        n_score_neg: int = 5,
        output_path: Optional[str] = None,
        skip_scoring: bool = False,
    ) -> Dict[int, Dict]:
        """
        Run the full explain → score pipeline for a set of feature IDs.

        Parameters
        ----------
        bank            : FeatureExemplarBank built from SAE activations
        feature_ids     : which features to explain
        n_score_pos     : held-out positives for scoring
        n_score_neg     : held-out negatives for scoring
        output_path     : if given, stream partial results to this JSON file
        skip_scoring    : if True, skip the scoring step (faster, no quality estimate)

        Returns
        -------
        results dict keyed by feature_id
        """
        n_tales = len(bank.texts)
        results: Dict[int, Dict] = {}
        feature_ids_list = list(feature_ids)

        logger.info(
            f"Running auto-interp on {len(feature_ids_list)} features "
            f"({'with' if not skip_scoring else 'without'} scoring)"
        )

        for fi, fid in enumerate(feature_ids_list):
            logger.info(f"  [{fi+1}/{len(feature_ids_list)}] Feature {fid}")
            stats = bank.feature_stats(fid)
            stats["n_total"] = n_tales

            positives, negatives = bank.get_exemplars(fid)

            # Generate explanation
            explanation = self.explain_feature(fid, positives, negatives, stats)

            entry: Dict = {
                "feature_id": fid,
                "explanation": explanation,
                "stats": stats,
                "n_pos_exemplars": len(positives),
                "positive_exemplars": [
                    {
                        "index": ex.index,
                        "title": ex.title,
                        "region": ex.region,
                        "activation": ex.activation,
                        "text": ex.text,
                    }
                    for ex in positives
                ],
                "score": None,
                "accuracy": None,
            }

            # Score explanation on held-out examples
            if not skip_scoring and stats["n_active"] >= (n_score_pos + 2):
                # Build held-out test set: sample pos and neg not seen in exemplars
                seen_pos = {ex.index for ex in positives}
                all_acts = bank.activations[:, fid]
                all_pos_idx = [i for i in np.where(all_acts > 0)[0] if i not in seen_pos]
                all_neg_idx = list(np.where(all_acts == 0)[0])

                rng = np.random.default_rng(42 + fid)
                chosen_pos = list(rng.choice(
                    all_pos_idx,
                    size=min(n_score_pos, len(all_pos_idx)),
                    replace=False,
                ))
                chosen_neg = list(rng.choice(
                    all_neg_idx,
                    size=min(n_score_neg, len(all_neg_idx)),
                    replace=False,
                ))

                test_exemplars = (
                    [bank._make_exemplar(i, float(all_acts[i])) for i in chosen_pos]
                    + [bank._make_exemplar(i, 0.0) for i in chosen_neg]
                )
                test_labels = [1] * len(chosen_pos) + [0] * len(chosen_neg)

                # Shuffle test set
                order = list(range(len(test_exemplars)))
                rng.shuffle(order)
                test_exemplars = [test_exemplars[i] for i in order]
                test_labels    = [test_labels[i]    for i in order]

                score_result = self.score_explanation(fid, explanation, test_exemplars, test_labels)
                entry.update(
                    score=score_result["score"],
                    accuracy=score_result["accuracy"],
                    precision=score_result.get("precision"),
                    recall=score_result.get("recall"),
                )

            results[fid] = entry

            # Checkpoint after each feature
            if output_path:
                _save_results(results, output_path)

        logger.info(
            f"Auto-interp complete. Explained {len(results)} features. "
            + (f"Saved to {output_path}" if output_path else "")
        )
        return results


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _save_results(results: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert int keys to strings for JSON
    serialisable = {str(k): v for k, v in results.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


def load_results(path: str) -> Dict[int, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Cluster-level auto-interp (no SAE needed) ─────────────────────────────────

def explain_clusters(
    cluster_assignments: np.ndarray,
    texts: List[str],
    metadata=None,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    n_pos: int = 6,
    n_neg: int = 4,
    excerpt_len: int = 400,
    output_path: Optional[str] = None,
    rate_limit_delay: float = 0.5,
) -> Dict[int, str]:
    """
    Convenience wrapper: generate one-sentence descriptions for k-means clusters
    without requiring an SAE.  Builds a fake binary activation matrix
    (1 = in cluster, 0 = not in cluster) and runs explain_feature per cluster.

    Returns dict mapping cluster_id → explanation string.
    """
    n_tales = len(texts)
    n_clusters = int(cluster_assignments.max()) + 1

    # Binary membership matrix: (n_tales, n_clusters)
    membership = np.zeros((n_tales, n_clusters), dtype=np.float32)
    for i, c in enumerate(cluster_assignments):
        membership[i, c] = 1.0

    bank = FeatureExemplarBank(
        activations=membership,
        texts=texts,
        metadata=metadata,
        n_pos=n_pos,
        n_neg=n_neg,
        excerpt_len=excerpt_len,
    )
    interp = AutoInterp(
        api_key=api_key,
        model=model,
        rate_limit_delay=rate_limit_delay,
    )
    results = interp.run(
        bank,
        feature_ids=range(n_clusters),
        skip_scoring=True,
        output_path=output_path,
    )
    return {fid: entry["explanation"] for fid, entry in results.items()}
