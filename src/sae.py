"""
Sparse Autoencoder (SAE) for decomposing folktale embeddings.

Implements the TopK SAE architecture from EleutherAI ("Scaling and evaluating
sparse autoencoders", Gao et al. 2024).  Each forward pass produces exactly k
non-zero activations per sample, giving a sparse code z with the representation
  x̂ = W_dec @ z + b_pre  (xDz+ in the project notation)

Architecture:
  encoder: Linear(d_model → n_features) + bias
  activation: TopK(k) applied to ReLU pre-activations
  decoder: Linear(n_features → d_model, no bias)
           columns kept at unit norm throughout training

Training loss: MSE reconstruction  +  auxiliary AuxK loss to revive dead features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder.

    Attributes
    ----------
    d_model    : input / output dimensionality (e.g. 384 for all-MiniLM-L6-v2)
    n_features : SAE hidden dimension (typically 4 × d_model)
    k          : number of active features per sample
    """

    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        # Pre-encoder bias (subtracted from input before encoding)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        # Encoder
        self.W_enc = nn.Linear(d_model, n_features, bias=True)

        # Decoder — no bias; b_pre is added back after decoding
        self.W_dec = nn.Linear(n_features, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_enc.bias)
        # Decoder columns start as unit vectors
        nn.init.normal_(self.W_dec.weight)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder columns onto the unit sphere."""
        self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x → sparse codes z.

        Returns
        -------
        z : (batch, n_features) sparse tensor with exactly k non-zeros per row
        """
        z_pre = self.W_enc(x - self.b_pre)           # (B, n_features)
        z_relu = F.relu(z_pre)                        # only positive activations

        # Keep only the k largest; zero the rest
        topk_vals, topk_idx = torch.topk(z_relu, self.k, dim=-1)
        z = torch.zeros_like(z_relu)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.W_dec(z) + self.b_pre

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction x̂, sparse codes z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ── Training ──────────────────────────────────────────────────────────────────

def train_sae(
    embeddings: np.ndarray,
    n_features: int = 1536,
    k: int = 16,
    n_epochs: int = 150,
    batch_size: int = 256,
    lr: float = 2e-4,
    auxk_alpha: float = 1 / 32,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[TopKSAE, Dict]:
    """
    Train a TopK SAE on pre-computed sentence-transformer embeddings.

    Parameters
    ----------
    embeddings      : (n_samples, d_model) float32 array
    n_features      : SAE hidden dim (default 4 × 384 = 1536)
    k               : active features per sample
    n_epochs        : training epochs
    batch_size      : mini-batch size
    lr              : Adam learning rate
    auxk_alpha      : weight for AuxK dead-feature revival loss
    device          : 'cpu' or 'cuda'
    checkpoint_path : if given, save the trained model here
    seed            : random seed

    Returns
    -------
    (sae, history)  where history contains per-epoch losses and dead-feature counts
    """
    torch.manual_seed(seed)
    d_model = embeddings.shape[1]
    n_samples = len(embeddings)
    logger.info(
        f"Training TopK SAE  d={d_model} → {n_features} features  k={k}  "
        f"epochs={n_epochs}  n={n_samples}"
    )

    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    sae = TopKSAE(d_model, n_features, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    # AuxK: an extra TopK pass with k_aux = n_features // 2 to revive dead features
    k_aux = min(n_features // 2, 512)

    history: Dict[str, List] = {
        "recon_loss": [],
        "auxk_loss": [],
        "total_loss": [],
        "dead_features": [],
    }

    n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
    activation_counts = torch.zeros(n_features, device=device)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        ep_recon = ep_aux = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = X[idx]

            # ── Forward ──────────────────────────────────────────────────────
            x_hat, z = sae(x)
            recon_loss = F.mse_loss(x_hat, x)

            # ── AuxK loss (revive dead features) ─────────────────────────────
            # Reconstruct residual using top-k_aux dead-feature directions
            with torch.no_grad():
                dead_mask = (activation_counts == 0)
            residual = (x - x_hat).detach()
            z_pre_full = F.relu(sae.W_enc(x - sae.b_pre))
            # Zero out live features to focus AuxK on dead ones
            z_dead = z_pre_full * dead_mask.float()
            topk_aux_vals, topk_aux_idx = torch.topk(z_dead, min(k_aux, int(dead_mask.sum().item()) or 1), dim=-1)
            z_aux = torch.zeros_like(z_dead)
            z_aux.scatter_(-1, topk_aux_idx, topk_aux_vals)
            aux_recon = sae.W_dec(z_aux)
            aux_loss = F.mse_loss(aux_recon, residual)

            total_loss = recon_loss + auxk_alpha * aux_loss

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            sae._normalize_decoder()

            ep_recon += recon_loss.item()
            ep_aux += aux_loss.item()

            # Track which features were active this batch
            with torch.no_grad():
                activation_counts += (z > 0).float().sum(0)

        # Reset counts each epoch so dead_features reflects the *current* epoch
        dead = int((activation_counts == 0).sum().item())
        activation_counts.zero_()

        ep_recon /= n_batches
        ep_aux /= n_batches
        history["recon_loss"].append(ep_recon)
        history["auxk_loss"].append(ep_aux)
        history["total_loss"].append(ep_recon + auxk_alpha * ep_aux)
        history["dead_features"].append(dead)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:4d}/{n_epochs}  recon={ep_recon:.5f}  "
                f"aux={ep_aux:.5f}  dead={dead}/{n_features}"
            )

    # Final variance-explained
    with torch.no_grad():
        sae.eval()
        x_hat_all, _ = sae(X)
        total_var = ((X - X.mean(0)) ** 2).sum().item()
        resid_var = ((X - x_hat_all) ** 2).sum().item()
        var_explained = 1.0 - resid_var / total_var
    logger.info(f"SAE training complete. Variance explained: {var_explained:.4f}")
    history["var_explained"] = var_explained

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d_model": d_model,
                "n_features": n_features,
                "k": k,
                "state_dict": sae.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )
        logger.info(f"SAE checkpoint saved → {checkpoint_path}")

    return sae, history


# ── Inference helpers ──────────────────────────────────────────────────────────

def load_sae(checkpoint_path: str, device: str = "cpu") -> TopKSAE:
    """Load a saved SAE from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sae = TopKSAE(ckpt["d_model"], ckpt["n_features"], ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    logger.info(
        f"Loaded SAE from {checkpoint_path}  "
        f"({ckpt['d_model']} → {ckpt['n_features']}, k={ckpt['k']})"
    )
    return sae


@torch.no_grad()
def get_sparse_activations(
    sae: TopKSAE,
    embeddings: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode a numpy embedding matrix through the SAE encoder.

    Returns
    -------
    z : (n_samples, n_features) float32 sparse activation array
    """
    X = torch.tensor(embeddings, dtype=torch.float32)
    parts = []
    sae.eval()
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size].to(device)
        z = sae.encode(batch)
        parts.append(z.cpu().numpy())
    return np.vstack(parts)
