"""Synthetic-batch verification for Task 3.3 IL loss machinery.

Goal: prove the BCE label construction + mean reduction + gradient flow can train
a MixtureWeightHead to recognize the correct anchor when GT trajectory equals
that anchor. We use a tiny dummy encoder (not the full Emu3Pi0) — verifying the
loss construction is correct end-to-end is enough; real-model convergence is
covered by Task 3.4 training curves.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.policy_head.mixture_weight_head import MixtureWeightHead


def _make_dummy_encoder(input_dim: int, hidden: int) -> nn.Module:
    """Trivial encoder: flatten input → 2-layer MLP → hidden."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
    )


def test_mixture_head_learns_correct_anchor():
    """Mirror Stage 2-A BCE setup: per-sample query (= GT trajectory) + N candidate
    anchors. Encoder input = concat(query, candidate). Head output high iff candidate
    matches query. Mirrors how Emu3Pi0.forward conditions mixture_logit on (vlm_h,
    anchor_token). Pass: for every sample, sigmoid at matching anchor > 0.7, others < 0.3.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    anchor_path = os.path.join(REPO_ROOT, "cache", "anchor_centers_N20.npy")
    anchors_phys = torch.from_numpy(np.load(anchor_path)).float()  # (20, 8, 3)
    N_a, N_f, action_dim = anchors_phys.shape
    assert (N_a, N_f, action_dim) == (20, 8, 3), f"unexpected anchor shape {anchors_phys.shape}"

    hidden = 64
    flat_dim = N_f * action_dim

    encoder = _make_dummy_encoder(input_dim=2 * flat_dim, hidden=hidden)
    head = MixtureWeightHead(action_hidden_size=hidden)
    nn.init.normal_(head.mlp[-1].weight, std=0.05)
    nn.init.zeros_(head.mlp[-1].bias)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=3e-3)

    B = 32
    n_steps = 400
    anc_flat = anchors_phys.view(N_a, flat_dim)                       # (N_a, flat_dim)

    for step in range(n_steps):
        # Per-sample target index: cycle through all anchors so every label balanced.
        targets = torch.randint(0, N_a, (B,))                          # (B,)
        query = anc_flat[targets]                                       # (B, flat_dim)

        # Build (B, N_a, flat_dim*2) = (query repeated, candidate)
        query_rep = query[:, None, :].expand(-1, N_a, -1)              # (B, N_a, flat_dim)
        cand_rep  = anc_flat[None, :, :].expand(B, -1, -1)             # (B, N_a, flat_dim)
        enc_in = torch.cat([query_rep, cand_rep], dim=-1)              # (B, N_a, 2*flat_dim)

        feats = encoder(enc_in.view(B * N_a, -1))                      # (B*N_a, hidden)
        logits = head(feats).view(B, N_a)                              # (B, N_a)

        labels = (torch.arange(N_a)[None, :] == targets[:, None]).to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- Eval: for every (sample, target) pair, sigmoid at target should dominate. ----
    with torch.no_grad():
        targets_eval = torch.arange(N_a)                                # one of each
        Be = N_a
        query_eval = anc_flat[targets_eval]
        query_rep = query_eval[:, None, :].expand(-1, N_a, -1)
        cand_rep  = anc_flat[None, :, :].expand(Be, -1, -1)
        enc_in = torch.cat([query_rep, cand_rep], dim=-1)
        logits_eval = head(encoder(enc_in.view(Be * N_a, -1))).view(Be, N_a)
        probs = torch.sigmoid(logits_eval)

    p_target = probs[torch.arange(Be), targets_eval]                    # (Be,)
    mask_other = ~(torch.arange(N_a)[None, :] == targets_eval[:, None])
    p_others_max = probs.where(mask_other, torch.zeros_like(probs)).amax(dim=1)

    print(f"\n[test] p(target) min/mean/max = {p_target.min():.4f} / {p_target.mean():.4f} / {p_target.max():.4f}")
    print(f"[test] p(other)  min/mean/max = {p_others_max.min():.4f} / {p_others_max.mean():.4f} / {p_others_max.max():.4f}")

    assert p_target.min().item() > 0.7, \
        f"Some target probability too low: min={p_target.min().item():.4f}"
    assert p_others_max.max().item() < 0.3, \
        f"Some negative probability too high: max={p_others_max.max().item():.4f}"


def test_bce_initial_value_with_mean_reduction():
    """Plan_3.md §1.1 vs ours: with mean reduction, initial BCE ≈ ln 2 ≈ 0.693
    (single anchor positive among N_a, all logits 0). With sum, would be ~13.86.
    """
    N_a = 20
    B = 4
    logits = torch.zeros(B, N_a)
    labels = torch.zeros(B, N_a)
    labels[:, 5] = 1.0

    bce_mean = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    bce_sum  = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")

    expected_mean = float(np.log(2.0))   # ≈ 0.6931
    expected_sum  = expected_mean * B * N_a

    assert abs(bce_mean.item() - expected_mean) < 1e-5, \
        f"mean BCE wrong: {bce_mean.item()} vs expected {expected_mean}"
    assert abs(bce_sum.item() - expected_sum) < 1e-3, \
        f"sum BCE wrong: {bce_sum.item()} vs expected {expected_sum}"

    # The whole point: mean ≪ sum → mean keeps BCE comparable to FM-IL initial 0.27
    assert bce_mean.item() < 1.0
    assert bce_sum.item() > 10.0


def test_chunked_bce_matches_unchunked():
    """Verify chunking (sub_idx slices → cat) yields the same loss as a single full pass."""
    torch.manual_seed(42)
    B, N_a = 4, 20
    k_plus = torch.tensor([3, 7, 12, 19])
    logits_full = torch.randn(B, N_a)
    labels_full = (torch.arange(N_a)[None, :] == k_plus[:, None]).float()

    bce_unchunked = F.binary_cross_entropy_with_logits(
        logits_full, labels_full, reduction="mean")

    chunk = 5
    pieces_l, pieces_lab = [], []
    for c_start in range(0, N_a, chunk):
        c_end = min(N_a, c_start + chunk)
        pieces_l.append(logits_full[:, c_start:c_end])
        pieces_lab.append(labels_full[:, c_start:c_end])
    logits_cat = torch.cat(pieces_l, dim=1)
    labels_cat = torch.cat(pieces_lab, dim=1)
    bce_chunked = F.binary_cross_entropy_with_logits(
        logits_cat, labels_cat, reduction="mean")

    assert torch.allclose(bce_unchunked, bce_chunked, atol=1e-7), \
        f"chunked != unchunked: {bce_unchunked.item()} vs {bce_chunked.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
