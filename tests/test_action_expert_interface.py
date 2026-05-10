"""
Task 2.1 — Action Expert 现有接口梳理 smoke test.

Verifies the current (anchor-free) action expert token sequence and shape
contract without loading any pretrained weights. All tests use random init
tiny models or direct module instantiation.

Interface summary (current, before Task 2.2 anchor injection):
  action_h = cat([state_token(B,1,1024), action_projector_out(B,8,1024)])
           = (B, 9, 1024)    ← action_seq_len=9

  decode:  action_decoder(final_action_hidden[:, 1:, :], tau_emb_exp)
           i.e. [:, 1:, :] skips the state_token, keeps 8 action frames

Task 2.2 change points (DO NOT EDIT HERE, edit modeling_emu3.py):
  - modeling_emu3.py:2084/2297  cat([state_token, anchor_token, action_hs])
  - modeling_emu3.py:2141/2328  [:, 1:, :] → [:, 2:, :]
  - action_seq_len auto-updates from tensor shape, no constant to change

Run:
    conda run -n drivevla python -m pytest tests/test_action_expert_interface.py -v -s
"""

import torch
import torch.nn as nn
import pytest

from reference.Emu3.emu3.mllm.modeling_emu3 import (
    ActionProjector,
    FinalLayer,
    SinusoidalPosEmb,
    Emu3Model,
    Emu3Pi0SharedLayer,
)
from reference.Emu3.emu3.mllm.configuration_emu3 import Emu3Config

# ---------------------------------------------------------------------------
# Constants matching the real ckpt (action_config.hidden_size=1024)
# ---------------------------------------------------------------------------
B = 2
ACTION_FRAMES = 8      # N_f
ACTION_DIM = 3         # (x_long, y_lat, heading)
ACTION_HIDDEN = 1024   # action_config.hidden_size
PRE_ACTION_FRAMES = 3
STATE_INPUT_DIM = PRE_ACTION_FRAMES * ACTION_DIM + 4  # 13


# ---------------------------------------------------------------------------
# Test 1: ActionProjector output shape
# ---------------------------------------------------------------------------

def test_action_projector_shape():
    """action_projector must map (B, N_f, 3) + tau_emb → (B, N_f, action_hidden)."""
    ap = ActionProjector(in_channels=ACTION_DIM, dim=ACTION_HIDDEN, action_frames=ACTION_FRAMES)
    tau_emb_mod = SinusoidalPosEmb(dim=ACTION_HIDDEN)

    noisy_action = torch.randn(B, ACTION_FRAMES, ACTION_DIM)
    tau_vals = torch.rand(B)
    tau_emb = tau_emb_mod(tau_vals)                            # (B, 1024)
    tau_emb_exp = tau_emb.unsqueeze(1).expand(-1, ACTION_FRAMES, -1)  # (B, 8, 1024)

    out = ap(noisy_action, tau_emb_exp)

    assert out.shape == (B, ACTION_FRAMES, ACTION_HIDDEN), \
        f"ActionProjector output: expected ({B}, {ACTION_FRAMES}, {ACTION_HIDDEN}), got {out.shape}"
    print(f"\n[PASS] ActionProjector output: {out.shape}")


# ---------------------------------------------------------------------------
# Test 2: state_projector output shape
# ---------------------------------------------------------------------------

def test_state_projector_shape():
    """state_projector must map (B, 13) → (B, 1, action_hidden)."""
    state_projector = nn.Sequential(
        nn.Linear(STATE_INPUT_DIM, ACTION_HIDDEN),
        nn.SiLU(),
        nn.Linear(ACTION_HIDDEN, ACTION_HIDDEN),
    )

    pre_action = torch.randn(B, PRE_ACTION_FRAMES, ACTION_DIM)
    cmd = torch.randn(B, 4)
    state_in = torch.cat([pre_action.flatten(1), cmd], dim=1)  # (B, 13)
    state_tok = state_projector(state_in).unsqueeze(1)          # (B, 1, 1024)

    assert state_tok.shape == (B, 1, ACTION_HIDDEN), \
        f"state_projector output: expected ({B}, 1, {ACTION_HIDDEN}), got {state_tok.shape}"
    print(f"\n[PASS] state_projector output: {state_tok.shape}")


# ---------------------------------------------------------------------------
# Test 3: full action token sequence = (B, 1+N_f, action_hidden) = (B, 9, 1024)
# ---------------------------------------------------------------------------

def test_action_token_sequence_shape():
    """Current (anchor-free) action sequence must be (B, 9, action_hidden)."""
    ap = ActionProjector(in_channels=ACTION_DIM, dim=ACTION_HIDDEN, action_frames=ACTION_FRAMES)
    tau_emb_mod = SinusoidalPosEmb(dim=ACTION_HIDDEN)
    state_projector = nn.Sequential(
        nn.Linear(STATE_INPUT_DIM, ACTION_HIDDEN),
        nn.SiLU(),
        nn.Linear(ACTION_HIDDEN, ACTION_HIDDEN),
    )

    noisy_action = torch.randn(B, ACTION_FRAMES, ACTION_DIM)
    tau_emb_exp = tau_emb_mod(torch.rand(B)).unsqueeze(1).expand(-1, ACTION_FRAMES, -1)

    action_hs_no_state = ap(noisy_action, tau_emb_exp)          # (B, 8, 1024)
    state_in = torch.cat([
        torch.randn(B, PRE_ACTION_FRAMES, ACTION_DIM).flatten(1),
        torch.randn(B, 4),
    ], dim=1)
    state_tok = state_projector(state_in).unsqueeze(1)           # (B, 1, 1024)

    action_seq = torch.cat([state_tok, action_hs_no_state], dim=1)
    action_seq_len = action_seq.shape[1]

    assert action_seq.shape == (B, 1 + ACTION_FRAMES, ACTION_HIDDEN), \
        f"Action sequence: expected ({B}, 9, {ACTION_HIDDEN}), got {action_seq.shape}"
    assert action_seq_len == 9, f"action_seq_len: expected 9, got {action_seq_len}"
    print(f"\n[PASS] action token sequence: {action_seq.shape}  (action_seq_len={action_seq_len})")

    # Document the Task 2.2 change: after anchor injection this will be (B, 10, 1024)
    # TODO (Task 2.2): action_seq = cat([state_tok, anchor_tok, action_hs]) → (B, 10, 1024)
    #   - action_seq_len auto-updates to 10 from .shape[1] — no constant to change
    #   - modeling_emu3.py:2141 and 2328: change [:, 1:, :] → [:, 2:, :]


# ---------------------------------------------------------------------------
# Test 4: decoder slice [:, 1:, :] skips state_token → (B, N_f, action_dim)
# ---------------------------------------------------------------------------

def test_decoder_slice_shape():
    """action_decoder(final_hidden[:, 1:, :], tau_emb) must give (B, N_f, action_dim)."""
    tau_emb_mod = SinusoidalPosEmb(dim=ACTION_HIDDEN)
    fl = FinalLayer(hidden_size=ACTION_HIDDEN, out_channels=ACTION_DIM)
    tau_emb_exp = tau_emb_mod(torch.rand(B)).unsqueeze(1).expand(-1, ACTION_FRAMES, -1)

    # Simulate final_action_hidden (B, 9, action_hidden)
    final_hidden = torch.randn(B, 1 + ACTION_FRAMES, ACTION_HIDDEN)

    velo = fl(final_hidden[:, 1:, :], tau_emb_exp)  # skip state_token

    assert velo.shape == (B, ACTION_FRAMES, ACTION_DIM), \
        f"Decoder output: expected ({B}, {ACTION_FRAMES}, {ACTION_DIM}), got {velo.shape}"
    print(f"\n[PASS] decoder slice output: {velo.shape}")

    # TODO (Task 2.2): After anchor_token at index=1, change to [:, 2:, :]
    # mixture_weight_head reads: final_hidden[:, 1:2, :]  (anchor_token position)


# ---------------------------------------------------------------------------
# Test 5: SharedLayer preserves (vlm_h, action_h) shapes for dynamic action_seq_len
# ---------------------------------------------------------------------------

def _make_tiny_shared_layer():
    """Return a SharedLayer built from two tiny Emu3Model decoder layers."""
    cfg = Emu3Config(
        vocab_size=184622,   # must be ≥ pad_token_id=151643
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    tiny_model = Emu3Model(cfg)
    vlm_layer = tiny_model.layers[0]
    act_layer = tiny_model.layers[1]
    return Emu3Pi0SharedLayer(vlm_layer, act_layer), cfg.hidden_size


@pytest.mark.parametrize("action_seq_len", [9, 10])  # 9=current, 10=after Task 2.2
def test_shared_layer_dynamic_action_seq_len(action_seq_len):
    """SharedLayer must work for action_seq_len=9 (current) and 10 (after Task 2.2).

    This confirms action_seq_len is fully dynamic — no hardcoded constant in SharedLayer.
    """
    shared, H = _make_tiny_shared_layer()

    VLM_S = 16
    vlm_h = torch.randn(B, VLM_S, H)
    act_h = torch.randn(B, action_seq_len, H)
    total = VLM_S + action_seq_len
    mask = torch.zeros(B, 1, total, total)
    pos_ids = torch.arange(VLM_S).unsqueeze(0).expand(B, -1)

    out_vlm, out_act = shared(vlm_h, act_h, pos_ids, mask, VLM_S, action_seq_len, B)

    assert out_vlm.shape == vlm_h.shape, \
        f"vlm_h shape changed: {vlm_h.shape} → {out_vlm.shape}"
    assert out_act.shape == act_h.shape, \
        f"act_h shape changed: {act_h.shape} → {out_act.shape}"
    print(f"\n[PASS] SharedLayer with action_seq_len={action_seq_len}: "
          f"vlm_h={out_vlm.shape}, act_h={out_act.shape}")
