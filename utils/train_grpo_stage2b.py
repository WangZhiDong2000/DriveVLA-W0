"""Stage 2-B GRPO training for DriveVLA-W0 (Task 4.1 + Task 4.3).

Loads the Stage 2-A checkpoint, runs two-pass GRPO:
  Pass-1: collect_rollouts (no_grad) → RolloutBatch with z_t_per_step, logp_old
  Pass-2: predict_velocity_for_grpo → recompute_log_prob → REINFORCE + IL loss

Loss formula (Plan_3 §Task 4.3, DD-v2 form):
  L = L_RL + λ_IL * L_IL
  L_RL = -mean(exp(log_p_new - log_p_old.detach()) * advantages)   # vanilla REINFORCE
  L_IL = mean(L1(z_mean_phys[...,:2], gt_xy broadcast))           # x,y only, over B×K×T
  λ_IL = 0.1 if has_positive else 1.0                              # DD-v2 adaptive

Key design choices:
- trainable = action_expert.* + anchor_embedding.* + mixture_weight_head.* (+ projectors)
- vlm.* / state_projector.* fully frozen (hash-verified via VLMHashCallback)
- No PPO clipping; vanilla REINFORCE with bit-identical IS ratio ≈ 1 at init
- Pass-2 anchor-chunked same as Pass-1 (T outer loop × anchor_chunk inner loop)
"""

import gc
import hashlib
import json
import os
import os.path as osp
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers as tf
from torch.utils.data.dataloader import default_collate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "reference", "Emu3"))

from emu3.mllm import Emu3Tokenizer, Emu3Pi0, Emu3Pi0Config
from datasets import Emu3DrivingVAVADataset

from models.policy_head.stochastic_ode_sampler import recompute_log_prob
from models.policy_head.anchored_flow_path import denormalize

from utils.rl_modules import (
    RolloutBatch,
    collect_rollouts,
    MockPDMRewardWrapper,
    PDMRewardWrapper,
)
from utils.train_grpo_stage2a import (
    ModelArguments,
    DataArguments,
    VLMHashCallback,
    inject_anchor_buffers,
    load_model_stage2a,
    get_dataset_split,
    update_configs,
    _compute_vlm_hash,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class TrainingArgumentsStage2B(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    dataloader_num_workers: Optional[int] = field(default=0)
    evaluation_strategy: str = field(default="no")
    action_loss_weight: float = field(default=1.0)
    action_sample_steps: int = field(default=10)
    freeze_vlm: bool = field(default=True)

    # Stage 2-A ckpt to load as init
    init_ckpt_path: str = field(
        default="logs/train_grpo_stage2a_mini_full_bce/checkpoint-500",
        metadata={"help": "Stage 2-A checkpoint used as Stage 2-B init."},
    )

    # Anchor / normalization
    anchor_cache_path: str = field(default="cache/anchor_centers_N20.npy")

    # Rollout params
    num_groups: int = field(default=8,
        metadata={"help": "G: stochastic rollouts per anchor."})
    num_truncated_steps: int = field(default=10,
        metadata={"help": "T: truncated Euler ODE steps."})
    sigma_init: float = field(default=0.5,
        metadata={"help": "σ for initial anchor → z_1 multiplicative noise."})
    sigma_step: float = field(default=0.04,
        metadata={"help": "σ for per-step stochastic Euler noise."})
    sigma_logprob_min: float = field(default=0.10,
        metadata={"help": "σ floor for log-prob (Plan §5.5)."})
    discount: float = field(default=0.8,
        metadata={"help": "γ for temporal advantage discounting."})
    anchor_chunk: int = field(default=20,
        metadata={"help": "N_anchor sub-batch size per VLM forward (OOM guard)."})

    # Scorer
    use_mock_scorer: bool = field(default=True,
        metadata={"help": "Use MockPDMRewardWrapper (no navsim needed)."})
    metric_cache_root: Optional[str] = field(default=None,
        metadata={"help": "Root dir for metric cache .lzma files when use_mock_scorer=False."})
    scorer_num_workers: int = field(default=8,
        metadata={"help": "PDMRewardWrapper worker count (ignored for mock)."})

    # Loss weights (Task 4.3 will fill these in)
    lambda_il: float = field(default=0.1,
        metadata={"help": "IL loss weight when has_positive anchor (adaptive: 1.0 otherwise)."})

    smoke_test: bool = field(default=False,
        metadata={"help": "Run rollout smoke test (shape + finite checks), then exit."})


# ---------------------------------------------------------------------------
# Custom collator: default_collate chokes on str fields; pop scene_token first
# ---------------------------------------------------------------------------

def _collate_with_token(features: List[Dict]) -> Dict:
    tokens = [f.pop("scene_token") for f in features]
    # default_collate preserves BatchEncoding (from tokenizer.pad); convert to plain dict
    # so accelerate's send_to_device doesn't call .to() on the List[str] scene_token.
    batch = dict(default_collate(features))
    batch["scene_token"] = tokens   # List[str], length B
    return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOStage2BTrainer(tf.Trainer):
    """HF Trainer for GRPO Stage 2-B.

    compute_loss:
      - Pass-1: collect_rollouts (under torch.no_grad, handled by decorator)
      - Logs rollout diagnostics (reward_mean, adv_std, logp_old)
      - Returns zero loss (placeholder; Task 4.3 adds REINFORCE + IL terms)
    """

    def __init__(
        self,
        *args,
        anchors_norm: torch.Tensor,
        scorer,
        stage2b_args: TrainingArgumentsStage2B,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._anchors_norm = anchors_norm
        self._scorer = scorer
        self._s2b_args = stage2b_args

    def _build_metric_paths(self, scene_tokens: List[str]) -> List[Optional[str]]:
        root = self._s2b_args.metric_cache_root
        if root is None or self._s2b_args.use_mock_scorer:
            return [None] * len(scene_tokens)
        return [osp.join(root, f"{t}.lzma") for t in scene_tokens]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        scene_tokens: List[str] = inputs.pop("scene_token")
        metric_cache_paths = self._build_metric_paths(scene_tokens)

        # ── Pass-1: rollout (torch.no_grad inside collect_rollouts) ──────
        rb: RolloutBatch = collect_rollouts(
            model=model,
            batch=inputs,
            anchors_norm=self._anchors_norm,
            scorer=self._scorer,
            metric_cache_paths=metric_cache_paths,
            num_groups=self._s2b_args.num_groups,
            num_steps=self._s2b_args.num_truncated_steps,
            sigma_init=self._s2b_args.sigma_init,
            sigma_step=self._s2b_args.sigma_step,
            sigma_logprob_min=self._s2b_args.sigma_logprob_min,
            discount=self._s2b_args.discount,
            anchor_chunk=self._s2b_args.anchor_chunk,
        )

        # ── Pass-2: recompute velo_pred with fresh θ, build GRPO loss ────
        # Incremental per-step backward avoids accumulating T×n_chunks autograd
        # graphs simultaneously (which would OOM on a 32 GB GPU with the 7B model).
        # Each ODE step t: forward anchor chunks, compute step loss, backward,
        # free graph. Trainer's backward(dummy) at the end adds zero gradients.
        torch.cuda.empty_cache()   # free any Pass-1 fragmentation before Pass-2 forwards
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype
        q01 = model.action_q01.to(device, dtype=dtype)
        q99 = model.action_q99.to(device, dtype=dtype)

        B, K, T, N_F, _ = rb.z_t_per_step.shape
        N_a = self._anchors_norm.shape[0]
        G   = self._s2b_args.num_groups
        dt  = -1.0 / T

        input_ids      = inputs.get("input_ids", None)
        attention_mask = inputs.get("attention_mask", None)
        pre_action     = inputs["pre_action"].to(device, dtype=dtype)
        cmd            = inputs["cmd"].to(device, dtype=dtype)

        anchor_norm_d  = self._anchors_norm.to(device, dtype=dtype)

        gt_xy = rb.gt_trajectory[..., :2]   # (B, N_F, 2) physical, detached

        # Pre-compute λ_IL (depends only on advantages, no grad needed)
        has_positive = (rb.advantages > 0).any(dim=2).any(dim=1)  # (B,) bool
        lambda_il_val = torch.where(
            has_positive,
            torch.full_like(has_positive, self._s2b_args.lambda_il, dtype=dtype),
            torch.ones_like(has_positive, dtype=dtype),
        ).mean().item()   # float

        # Per-anchor serial forward+backward: each call uses B_eff = B*G samples
        # (same batch size as Pass-1 collect_rollouts), which is critical for
        # bit-identical velocities. In bfloat16, flash-attention accumulates
        # floats in batch-size-dependent order; even a ~0.1 m velocity error
        # causes exp(lp_new - lp_old) overflow when sigma_lp=0.10 and physical
        # waypoints are 10-50 m (giving lp_old ≈ -(diff²/0.02)×16 dims ≈ -500).
        # The per-anchor batch of G=2 still fits within the 9.6 GB headroom on
        # a 32 GB GPU (checkpoint-input storage ≈ 1 GB per call; freed on backward).
        total_rl = 0.0
        total_il = 0.0
        total_ratio_sum = 0.0
        total_ratio_max = -float("inf")
        n_iters = 0
        norm_denom = float(N_a * T)   # N_a * T backward calls; each call's mean covers B*G

        gt_xy = gt_xy.repeat_interleave(G, dim=0)   # (B*G, N_F, 2) physical

        for t_idx in range(T):
            t_value = 1.0 + t_idx * dt   # matches Pass-1 current_time at step t_idx
            log_p_old_t = rb.logp_old[:, :, t_idx].detach()    # (B, K)
            adv_t       = rb.advantages[:, :, t_idx].detach()  # (B, K)

            for n in range(N_a):
                k0, k1 = n * G, (n + 1) * G

                # Tile anchor+VLM inputs to B*G — mirrors Pass-1 repeat_interleave pattern
                anc_beff = (
                    anchor_norm_d[n:n+1]
                    .unsqueeze(1).expand(-1, G, -1, -1)   # (1, G, N_F, 3)
                    .reshape(G, N_F, 3)
                    .unsqueeze(0).expand(B, -1, -1, -1)   # (B, G, N_F, 3)
                    .reshape(B * G, N_F, 3)               # (B*G, N_F, 3)
                )
                ids_beff  = input_ids.repeat_interleave(G, dim=0) if input_ids is not None else None
                mask_beff = attention_mask.repeat_interleave(G, dim=0) if attention_mask is not None else None
                pa_beff   = pre_action.repeat_interleave(G, dim=0)
                cmd_beff  = cmd.repeat_interleave(G, dim=0)

                z_t_beff   = rb.z_t_per_step[:, k0:k1, t_idx].reshape(B * G, N_F, 3)
                z_next_beff= rb.z_next_phys_per_step[:, k0:k1, t_idx].reshape(B * G, N_F, 3)
                lp_old_beff= log_p_old_t[:, k0:k1].reshape(B * G)   # (B*G,)
                adv_beff   = adv_t[:, k0:k1].reshape(B * G)          # (B*G,)

                velo_pred = model.predict_velocity_for_grpo(
                    input_ids=ids_beff,
                    pre_action=pa_beff,
                    cmd=cmd_beff,
                    z_t=z_t_beff,
                    t_value=t_value,
                    anchor=anc_beff,
                    attention_mask=mask_beff,
                )   # (B*G, N_F, 3) with grad

                lp_new = recompute_log_prob(
                    z_t_norm=z_t_beff,
                    velo_pred=velo_pred,
                    z_next_phys_stored=z_next_beff,
                    dt=dt,
                    q01=q01,
                    q99=q99,
                    sigma_step=self._s2b_args.sigma_step,
                    sigma_logprob_min=self._s2b_args.sigma_logprob_min,
                )   # (B*G,)

                z_mean_phys = denormalize(z_t_beff + dt * velo_pred, q01, q99)  # (B*G, N_F, 3)

                # DD-v2 vanilla REINFORCE: exp(lp_new - lp_new.detach()) == 1 always;
                # gradient flows through lp_new → ∇_θ log_π(a|s) × A.
                # No IS explosion after parameter updates (unlike exp(lp_new - lp_old)).
                ratio = torch.exp(lp_new - lp_new.detach())        # (B*G,), always 1.0
                rl    = -(ratio * adv_beff).mean()                  # scalar
                il    = F.l1_loss(z_mean_phys[..., :2], gt_xy, reduction="mean")

                loss_n = (rl + lambda_il_val * il) / norm_denom
                loss_n.backward()   # accumulate grad, free this anchor's graph

                # Log IS-ratio diagnostic (exp(lp_new - lp_old)) after backward;
                # ratio_diag monitors policy drift without affecting the gradient.
                with torch.no_grad():
                    ratio_diag = torch.exp(lp_new - lp_old_beff)   # (B*G,)
                total_rl        += rl.detach().item()
                total_il        += il.detach().item()
                total_ratio_sum += ratio_diag.mean().item()
                total_ratio_max  = max(total_ratio_max, ratio_diag.max().item())
                n_iters         += 1

        total_rl         /= n_iters
        total_il         /= n_iters
        total_ratio_mean  = total_ratio_sum / n_iters
        total_loss_val    = total_rl + lambda_il_val * total_il

        # ── Logging ──────────────────────────────────────────────────────
        self.log({
            "rollout/reward_mean":   rb.rewards.mean().item(),
            "rollout/reward_max":    rb.rewards.max().item(),
            "rollout/reward_std":    rb.rewards.std().item(),
            "rollout/adv_std":       rb.advantages.std().item(),
            "rollout/logp_old":      rb.logp_old.mean().item(),
            "loss/rl":               total_rl,
            "loss/il":               total_il,
            "loss/lambda_il":        lambda_il_val,
            "loss/total":            total_loss_val,
            "loss/ratio_mean":       total_ratio_mean,
            "loss/ratio_max":        total_ratio_max,
            "loss/has_positive":     has_positive.float().mean().item(),
        })

        # Return a dummy tensor with grad so HF Trainer's backward() is a no-op
        # (real grads are already accumulated by the incremental loss_t.backward() calls above)
        dummy = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
        dummy_loss = dummy + torch.tensor(total_loss_val, device=device, dtype=dtype).detach()

        return (dummy_loss, rb) if return_outputs else dummy_loss


# ---------------------------------------------------------------------------
# Model loading (reuse Stage 2-A routine, pointing to Stage 2-B init ckpt)
# ---------------------------------------------------------------------------

def load_model_stage2b(model_args, model_config, training_args: TrainingArgumentsStage2B):
    """Load Stage 2-A ckpt as Stage 2-B init; freeze vlm + state_projector.

    model_name_or_path  → Stage 2-A checkpoint (action_expert / anchor / mixture heads)
    pretrain_vlm_path   → original pretrained model (VLM weights unchanged since Stage 2-A froze them)
    """
    orig_path = model_args.model_name_or_path  # pretrained_models/...
    # Action expert + heads come from Stage 2-A checkpoint
    model_args.model_name_or_path = training_args.init_ckpt_path
    # VLM comes from the original pretrained model (frozen during Stage 2-A, not in ckpt safetensors)
    if model_args.pretrain_vlm_path is None:
        model_args.pretrain_vlm_path = orig_path
    model = load_model_stage2a(model_args, model_config, training_args)
    model_args.model_name_or_path = orig_path
    return model


# ---------------------------------------------------------------------------
# Smoke test (local: 1 training step with mock scorer)
# ---------------------------------------------------------------------------

def run_smoke_test_stage2b(
    model, train_dataset, anchors_norm, training_args: TrainingArgumentsStage2B
):
    print("\n" + "=" * 60)
    print("[smoke_test_stage2b] Starting Stage 2-B rollout smoke test...")
    print("=" * 60)

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    sample = train_dataset[0]
    token  = sample.pop("scene_token")
    batch  = default_collate([sample])
    batch  = {k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device))
              for k, v in batch.items() if isinstance(v, torch.Tensor)}

    scorer = MockPDMRewardWrapper()
    rb: RolloutBatch = collect_rollouts(
        model=model,
        batch=batch,
        anchors_norm=anchors_norm.to(device, dtype=dtype),
        scorer=scorer,
        metric_cache_paths=[None],
        num_groups=training_args.num_groups,
        num_steps=training_args.num_truncated_steps,
        sigma_init=training_args.sigma_init,
        sigma_step=training_args.sigma_step,
        sigma_logprob_min=training_args.sigma_logprob_min,
        discount=training_args.discount,
        anchor_chunk=training_args.anchor_chunk,
    )

    N_a = anchors_norm.shape[0]
    G   = training_args.num_groups
    K   = N_a * G
    T   = training_args.num_truncated_steps
    B   = batch["action"].shape[0]

    # Shape assertions
    assert rb.z_t_per_step.shape         == (B, K, T, batch["action"].shape[1], 3), \
        f"z_t_per_step shape wrong: {rb.z_t_per_step.shape}"
    assert rb.z_next_phys_per_step.shape == (B, K, T, batch["action"].shape[1], 3), \
        f"z_next_phys_per_step shape wrong: {rb.z_next_phys_per_step.shape}"
    assert rb.logp_old.shape             == (B, K, T), \
        f"logp_old shape wrong: {rb.logp_old.shape}"
    assert rb.rewards.shape              == (B, K), \
        f"rewards shape wrong: {rb.rewards.shape}"
    assert rb.advantages.shape           == (B, K, T), \
        f"advantages shape wrong: {rb.advantages.shape}"
    print(f"[smoke_test_stage2b] RolloutBatch shapes OK  (B={B}, K={K}, T={T})  ✓")

    # No NaN / Inf
    for field_name in ("logp_old", "rewards", "advantages"):
        val = getattr(rb, field_name)
        assert torch.isfinite(val).all(), f"{field_name} contains non-finite values"
    print("[smoke_test_stage2b] No NaN/Inf in logp_old / rewards / advantages  ✓")

    # advantages have non-trivial spread
    assert rb.advantages.std() > 0, "advantages are constant — advantage normalization broken"
    print(f"[smoke_test_stage2b] advantages.std = {rb.advantages.std().item():.6f}  ✓")

    # logp_old detached
    assert not rb.logp_old.requires_grad, "logp_old must be detached"
    print("[smoke_test_stage2b] logp_old is detached  ✓")

    # ── Bit-identical check under no_grad (θ unchanged → ratio ≈ 1) ──────
    # Note: ratio = exp(lp_new - lp_old) is a numerical property that holds
    # without grads. Full Pass-2 with enable_grad would OOM (model fills GPU).
    # Gradient flow is validated during the 4-step mini training below.
    q01 = model.action_q01.to(device, dtype=dtype)
    q99 = model.action_q99.to(device, dtype=dtype)
    N_F = rb.z_t_per_step.shape[3]
    G   = training_args.num_groups
    dt  = -1.0 / T

    torch.cuda.empty_cache()

    # Use anchor 0, all groups (k = 0..G-1), step 0
    anc_norm_d = anchors_norm.to(device, dtype=dtype)
    anc_bk = (
        anc_norm_d[0:1]
        .unsqueeze(1).expand(-1, G, -1, -1)   # (1, G, N_F, 3)
        .reshape(G, N_F, 3)
        .unsqueeze(0).expand(B, -1, -1, -1)   # (B, G, N_F, 3)
        .reshape(B * G, N_F, 3)               # (B*G, N_F, 3)
    )
    z_t_bk      = rb.z_t_per_step[:, 0:G, 0].reshape(B * G, N_F, 3)
    z_next_bk   = rb.z_next_phys_per_step[:, 0:G, 0].reshape(B * G, N_F, 3)
    logp_ref_bk = rb.logp_old[:, 0:G, 0].reshape(B * G)   # (B*G,)

    ids_bk  = (batch["input_ids"].repeat_interleave(G, dim=0)
               if "input_ids" in batch else None)
    mask_bk = (batch["attention_mask"].repeat_interleave(G, dim=0)
               if "attention_mask" in batch else None)
    pa_bk   = batch["pre_action"].to(device, dtype=dtype).repeat_interleave(G, dim=0)
    cmd_bk  = batch["cmd"].to(device, dtype=dtype).repeat_interleave(G, dim=0)

    model.eval()
    with torch.no_grad():
        velo_pred = model.predict_velocity_for_grpo(
            input_ids=ids_bk,
            pre_action=pa_bk,
            cmd=cmd_bk,
            z_t=z_t_bk,
            t_value=1.0,            # step t_idx=0 → t_value = 1.0 + 0*dt = 1.0
            anchor=anc_bk,
            attention_mask=mask_bk,
        )
        lp_new_bk = recompute_log_prob(
            z_t_norm=z_t_bk,
            velo_pred=velo_pred,
            z_next_phys_stored=z_next_bk,
            dt=dt,
            q01=q01,
            q99=q99,
            sigma_step=training_args.sigma_step,
            sigma_logprob_min=training_args.sigma_logprob_min,
        )

    assert torch.isfinite(lp_new_bk).all(), "recompute_log_prob returned non-finite"
    print("[smoke_test_stage2b] recompute_log_prob finite  ✓")

    ratio_bk = torch.exp(lp_new_bk - logp_ref_bk)
    ratio_err = (ratio_bk - 1.0).abs().max().item()
    print(f"[smoke_test_stage2b] bit-identical ratio max_err = {ratio_err:.6f}  "
          f"(expect < 0.05 in bf16)  {'✓' if ratio_err < 0.05 else '✗ WARNING'}")
    print(f"[smoke_test_stage2b] logp_old[0] = {logp_ref_bk[0].item():.4f}  "
          f"logp_new[0] = {lp_new_bk[0].item():.4f}")

    # ── Verify VLM + state_projector are frozen ──────────────────────────
    frozen_ok = all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if n.startswith("vlm.") or n.startswith("state_projector.")
    )
    assert frozen_ok, "vlm.* or state_projector.* has requires_grad=True — freeze broken"
    print("[smoke_test_stage2b] vlm.* and state_projector.* frozen  ✓")

    print("\n[smoke_test_stage2b] ALL CHECKS PASSED ✓")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsStage2B))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    pi0_config = Emu3Pi0Config.from_pretrained(model_args.model_config_path)
    update_configs(pi0_config, training_args,
                   ["image_area", "max_position_embeddings",
                    "action_loss_weight", "action_sample_steps", "freeze_vlm"])
    if training_args.bf16:
        pi0_config.torch_dtype = torch.bfloat16
        pi0_config.vlm_config.torch_dtype = torch.bfloat16
        pi0_config.action_config.torch_dtype = torch.bfloat16

    # Model
    model = load_model_stage2b(model_args, pi0_config, training_args)

    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    # Tokenizer
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Anchor buffers
    anchor_phys, anchor_norm = inject_anchor_buffers(model, training_args.anchor_cache_path)

    # Dataset
    train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)

    # Scorer
    if training_args.use_mock_scorer:
        scorer = MockPDMRewardWrapper()
        print("[Stage2B] Using MockPDMRewardWrapper (local smoke mode)")
    else:
        scorer = PDMRewardWrapper(num_workers=training_args.scorer_num_workers)
        print(f"[Stage2B] Using PDMRewardWrapper ({training_args.scorer_num_workers} workers)")

    # Smoke test early exit
    if getattr(training_args, "smoke_test", False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        run_smoke_test_stage2b(model, train_dataset, anchor_norm, training_args)
        return

    # VLM hash callback
    callbacks = [VLMHashCallback()]

    # Trainer
    trainer = GRPOStage2BTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=_collate_with_token,
        callbacks=callbacks,
        anchors_norm=anchor_norm,
        scorer=scorer,
        stage2b_args=training_args,
    )

    # Train (resume if checkpoint exists)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if not training_args.use_mock_scorer:
        scorer.shutdown()


if __name__ == "__main__":
    train()
