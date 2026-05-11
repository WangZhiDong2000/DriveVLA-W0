"""Stage 2-A IL adaptation training for DriveVLA-W0 anchored Flow Matching.

Trains action_expert.* + anchor_embedding.* + mixture_weight_head.* on top of the
87.2 PDMS ckpt with anchored flow paths (L_FM-IL on k+ only) and optional BCE loss.

Key differences from train_pi0.py:
- Always uses from_pretrained (init_fresh_expert=False is the only valid mode here)
- state_projector.* also frozen (preserves cmd understanding)
- Anchors loaded from cache/anchor_centers_N20.npy and injected per batch
- lambda_a / sigma_anchor warmup for R9 cold-start mitigation (Task 3.2)
- Custom GRPOStage2ATrainer.compute_loss injects anchor + warmup coefficients
- --smoke_test mode: 1 forward+backward, param hash check, no trainer.train()
- --bce_strategy: 'none' (default, mini smoke) | 'k_plus_neg' | 'full' (server)
"""

import gc
import hashlib
import json
import random
import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from torch.utils.data.dataloader import default_collate
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "reference", "Emu3"))

from emu3.mllm import Emu3Tokenizer, Emu3Pi0, Emu3Pi0Config
from datasets import Emu3DrivingVAVADataset


# ---------------------------------------------------------------------------
# Argument dataclasses (inherit patterns from train_pi0.py)
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")
    pretrain_vlm_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=1)
    VL: bool = field(default=False)
    actions: bool = field(default=True)
    actions_format: str = field(default="fast")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    driving: bool = field(default=True)
    use_previous_actions: bool = field(default=True)
    use_only_lidar: bool = field(default=False)
    use_lidar_and_image: bool = field(default=False)
    use_flip: bool = field(default=False)
    cur_frame_idx: int = field(default=3)
    action_dim: int = field(default=3)
    pre_action_frames: int = field(default=3)
    normalizer_path: Optional[str] = field(default=None)


@dataclass
class TrainingArgumentsStage2A(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    dataloader_num_workers: Optional[int] = field(default=0)
    evaluation_strategy: str = field(default="no")
    eval_steps: Optional[int] = field(default=None)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    eval_accumulation_steps: Optional[int] = field(default=1)
    action_loss_weight: float = field(default=1.0)
    action_sample_steps: int = field(default=10)
    freeze_vlm: bool = field(default=True)  # always True for Stage 2-A
    # Stage 2-A specific
    anchor_cache_path: str = field(default="cache/anchor_centers_N20.npy")
    anchor_warmup_steps: int = field(default=500,
        metadata={"help": "Steps to linearly warm up lambda_a 0→1 and sigma_anchor 0→0.04. "
                           "Use 50 for mini smoke, 500 for server full."})
    sigma_anchor_max: float = field(default=0.04,
        metadata={"help": "Max sigma_anchor after warmup (multiplicative noise std)."})
    bce_strategy: str = field(default="none",
        metadata={"help": "'none': FM-IL only (default, mini smoke); "
                           "'k_plus_neg': k+ + N random negatives; "
                           "'full': all N_anchor (server only)."})
    bce_n_neg: int = field(default=2,
        metadata={"help": "Number of negative anchors per sample when bce_strategy='k_plus_neg'."})
    bce_loss_weight: float = field(default=0.5,
        metadata={"help": "Weight on L_BCE in L_IL = L_FM-IL + w * L_BCE. Default 0.5 keeps "
                           "BCE comparable to FM-IL across training (FM-IL goes 0.27→0.027, "
                           "BCE stays ~0.5)."})
    bce_chunk_size: int = field(default=5,
        metadata={"help": "Anchors per BCE forward pass. With B=4, chunk=5 → effective batch 20. "
                           "Reduce to 2-4 if OOM on smaller GPUs."})
    smoke_test: bool = field(default=False,
        metadata={"help": "Run 1 forward+backward, print param/hash diagnostics, exit without training."})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_vlm_hash(model: Emu3Pi0) -> str:
    """SHA-256 of first 4 + last 4 layers' weight tensors + lm_head weight."""
    h = hashlib.sha256()
    layers = list(model.vlm.model.layers)
    sampled = layers[:4] + layers[-4:]
    for layer in sampled:
        for p in layer.parameters():
            h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    for p in model.vlm.lm_head.parameters():
        h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def select_positive_anchor(
    anchor_centers_phys: torch.Tensor,   # (N_anchor, N_f, 3) physical (m/rad)
    gt_action_norm: torch.Tensor,         # (B, N_f, 3) normalized [-1,1]
    q01: torch.Tensor,                    # (3,)
    q99: torch.Tensor,                    # (3,)
) -> torch.Tensor:
    """Return k+_idx (B,) long: argmin over (x,y) L2 in physical space (Plan §1.1)."""
    gt_phys = (gt_action_norm + 1.0) / 2.0 * (q99 - q01) + q01   # (B, N_f, 3)
    # distances over (x,y) only, summed across waypoints
    diffs = anchor_centers_phys[None, :, :, :2] - gt_phys[:, None, :, :2]  # (B, N_a, N_f, 2)
    dists = (diffs ** 2).sum(dim=(2, 3))                                      # (B, N_a)
    return dists.argmin(dim=1)                                                # (B,)


class WarmupScheduler:
    """Linear warmup for lambda_a and sigma_anchor (Task 3.2)."""

    def __init__(self, warmup_steps: int, sigma_anchor_max: float = 0.04):
        self.warmup_steps = max(1, warmup_steps)
        self.sigma_anchor_max = sigma_anchor_max

    def get(self, step: int):
        t = min(1.0, step / self.warmup_steps)
        return t, t * self.sigma_anchor_max   # lambda_a, sigma_anchor


class VLMHashCallback(tf.TrainerCallback):
    """Assert vlm.* parameters unchanged throughout training (Task 3.1 pass criteria)."""

    def __init__(self):
        self._initial_hash: Optional[str] = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._initial_hash = _compute_vlm_hash(model)
        print(f"[VLMHashCallback] Initial vlm hash: {self._initial_hash[:16]}...")

    def on_save(self, args, state, control, model=None, **kwargs):
        current = _compute_vlm_hash(model)
        ok = current == self._initial_hash
        print(f"[VLMHashCallback] step={state.global_step} hash_match={ok}")
        if not ok:
            raise RuntimeError("VLM parameters changed during Stage 2-A training! "
                               f"Expected {self._initial_hash[:16]}, got {current[:16]}")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        current = _compute_vlm_hash(model)
        ok = current == self._initial_hash
        print(f"[VLMHashCallback] on_train_end hash_match={ok}")
        if not ok:
            raise RuntimeError("VLM parameters changed during Stage 2-A training!")


# ---------------------------------------------------------------------------
# Trainer with custom compute_loss
# ---------------------------------------------------------------------------

class GRPOStage2ATrainer(tf.Trainer):
    """HF Trainer subclass that injects anchor + warmup coefficients into model.forward."""

    def __init__(self, *args, warmup_scheduler: WarmupScheduler,
                 anchor_centers_phys: torch.Tensor,   # (N_a, N_f, 3) physical
                 anchor_centers_norm: torch.Tensor,   # (N_a, N_f, 3) normalized
                 bce_strategy: str = "none",
                 bce_n_neg: int = 2,
                 bce_loss_weight: float = 0.5,
                 bce_chunk_size: int = 5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup = warmup_scheduler
        self._anchors_phys = anchor_centers_phys   # will be moved to device in compute_loss
        self._anchors_norm = anchor_centers_norm
        self._bce_strategy = bce_strategy
        self._bce_n_neg = bce_n_neg
        self._bce_loss_weight = bce_loss_weight
        self._bce_chunk_size = bce_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        step = self.state.global_step if self.state else 0
        lambda_a, sigma_anchor = self._warmup.get(step)

        gt_action = inputs["action"]                          # (B, N_f, 3) normalized
        device = gt_action.device

        anchors_phys = self._anchors_phys.to(device, dtype=gt_action.dtype)  # (N_a, N_f, 3)
        anchors_norm = self._anchors_norm.to(device, dtype=gt_action.dtype)

        q01 = model.action_q01.to(device, dtype=gt_action.dtype)  # (3,)
        q99 = model.action_q99.to(device, dtype=gt_action.dtype)

        # ---------- k+ selection ----------
        k_plus = select_positive_anchor(anchors_phys, gt_action, q01, q99)  # (B,)
        anchor_for_batch = anchors_norm[k_plus]   # (B, N_f, 3) normalized

        # ---------- FM-IL forward on k+ ----------
        fm_inputs = dict(inputs)
        fm_inputs["anchor"] = anchor_for_batch

        outputs = model(
            **fm_inputs,
            sigma_anchor=sigma_anchor,
            lambda_a=lambda_a,
        )
        loss = outputs.loss   # L_FM-IL(k+ only)

        # ---------- Optional BCE loss ----------
        if self._bce_strategy != "none":
            bce_loss = self._compute_bce_loss(
                model, inputs, anchors_norm, k_plus,
                sigma_anchor, lambda_a, device, gt_action.dtype,
            )
            fm_loss_value = loss.detach().float().item()
            loss = loss + self._bce_loss_weight * bce_loss
            self.log({"l_fm_il": fm_loss_value,
                      "l_bce": bce_loss.detach().float().item()})

        return (loss, outputs) if return_outputs else loss

    def _compute_bce_loss(
        self, model, inputs, anchors_norm, k_plus,
        sigma_anchor, lambda_a, device, dtype,
    ):
        """Compute L_BCE over all (or k+_neg subset) of N_anchor via batched chunked forwards.

        Returns: scalar tensor (mean reduction over (B, N_a_subset)).
        Plan_3.md §1.1 specifies sum, but mean keeps L_BCE ≈ 0.69 initial vs L_FM-IL ≈ 0.27,
        avoiding gradient domination by BCE. See plan glimmering-nibbling-kahn.md.
        """
        N_a = anchors_norm.shape[0]
        B = k_plus.shape[0]

        if self._bce_strategy == "full":
            anchor_indices = torch.arange(N_a, device=device)              # (N_a,)
        else:  # k_plus_neg
            idx_per_sample = []
            for b in range(B):
                kp = k_plus[b].item()
                neg_pool = [i for i in range(N_a) if i != kp]
                negs = random.sample(neg_pool, min(self._bce_n_neg, len(neg_pool)))
                idx_per_sample.append([kp] + negs)
            anchor_indices = torch.tensor(idx_per_sample, device=device)   # (B, n_neg+1)

        # full: anchor_indices is (N_a,) shared across batch → tile to (B, N_a)
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices[None, :].expand(B, -1).contiguous()  # (B, N_a)

        K = anchor_indices.shape[1]
        chunk = max(1, self._bce_chunk_size)
        all_logits, all_labels = [], []
        for c_start in range(0, K, chunk):
            c_end = min(K, c_start + chunk)
            sub_idx = anchor_indices[:, c_start:c_end]                     # (B, c)
            c_size = sub_idx.shape[1]

            # Tile inputs along batch dim by c_size: (B*c, ...). Only Tensor fields.
            anc_inputs = {k: v.repeat_interleave(c_size, dim=0)
                          for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            # anchor for tiled batch: gather (B*c, N_f, 3) — interleave matches sub_idx flatten
            anc_inputs["anchor"] = anchors_norm[sub_idx.reshape(-1)]

            out = model(**anc_inputs, sigma_anchor=sigma_anchor, lambda_a=lambda_a)
            logits = out.mixture_logit.view(B, c_size)                     # (B, c)
            labels = (sub_idx == k_plus[:, None]).to(logits.dtype)         # (B, c)
            all_logits.append(logits)
            all_labels.append(labels)

        logits_all = torch.cat(all_logits, dim=1)   # (B, K)
        labels_all = torch.cat(all_labels, dim=1)   # (B, K)
        return F.binary_cross_entropy_with_logits(logits_all, labels_all, reduction="mean")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_stage2a(model_args, model_config, training_args):
    """Load from_pretrained (never fresh expert), freeze vlm.* + state_projector.*."""
    model, loading_info = Emu3Pi0.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        pretrain_vlm_path=model_args.pretrain_vlm_path or model_args.model_name_or_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_loading_info=True,
    )

    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    # VLM keys appear in missing_keys because HF from_pretrained can't match vlm.*-prefixed
    # checkpoint keys to the nested Emu3MoE submodule; they are loaded separately via
    # pretrain_vlm_path. This matches train_pi0.py behavior — just print, don't fail on missing.
    new_head_missing = [k for k in missing if k.startswith(("anchor_embedding.", "mixture_weight_head."))]
    print(f"[Stage2A] missing_keys (total {len(missing)}): "
          f"{len(new_head_missing)} new-head keys + {len(missing)-len(new_head_missing)} other (expected vlm.* + action_expert.*)")
    print(f"[Stage2A] new-head missing: {new_head_missing}")
    print(f"[Stage2A] unexpected_keys = {unexpected}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in ckpt (ckpt mismatch?): {unexpected}")

    # Freeze vlm.* (already handled by model.freeze_vlm if freeze_vlm=True)
    model.freeze_vlm()

    # Additionally freeze state_projector.* (preserve cmd understanding, Plan §D7)
    for p in model.state_projector.parameters():
        p.requires_grad = False
    print("[Stage2A] Frozen: vlm.*, state_projector.*")

    # Diagnostic: count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Stage2A] Trainable: {trainable:,}  /  Total: {total:,}")

    # Verify vlm.* and state_projector.* truly have no grad
    vlm_grad_sum = sum(1 for p in model.vlm.parameters() if p.requires_grad)
    sp_grad_sum  = sum(1 for p in model.state_projector.parameters() if p.requires_grad)
    assert vlm_grad_sum == 0, f"vlm still has {vlm_grad_sum} trainable params!"
    assert sp_grad_sum  == 0, f"state_projector still has {sp_grad_sum} trainable params!"

    return model


# ---------------------------------------------------------------------------
# Anchor buffer injection
# ---------------------------------------------------------------------------

def inject_anchor_buffers(model: Emu3Pi0, anchor_cache_path: str):
    """Register anchor_centers_{phys,norm} as non-persistent buffers on the model."""
    anchor_phys = torch.from_numpy(np.load(anchor_cache_path)).float()  # (N_a, N_f, 3) physical
    q01 = model.action_q01.float()   # (3,)
    q99 = model.action_q99.float()   # (3,)
    anchor_norm = (anchor_phys - q01) / (q99 - q01) * 2.0 - 1.0        # (N_a, N_f, 3) normalized
    model.register_buffer("anchor_centers_phys", anchor_phys, persistent=False)
    model.register_buffer("anchor_centers_norm", anchor_norm, persistent=False)
    print(f"[Stage2A] Loaded {anchor_phys.shape[0]} anchors from {anchor_cache_path}")
    return anchor_phys, anchor_norm


# ---------------------------------------------------------------------------
# Dataset helpers (mirroring train_pi0.py)
# ---------------------------------------------------------------------------

def get_dataset_split(data_args, tokenizer):
    full_dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def update_configs(model_config, args, fields):
    def cross_update(a, b, field_name):
        if getattr(b, field_name, None) is None:
            setattr(b, field_name, getattr(a, field_name))
        else:
            setattr(a, field_name, getattr(b, field_name))
    for f in fields:
        cross_update(model_config, args, f)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(model, train_dataset, anchor_phys, anchor_norm, training_args):
    """1 forward+backward, check params/hash. Exit 0 on success."""
    print("\n" + "="*60)
    print("[smoke_test] Starting Stage 2-A smoke test...")
    print("="*60)

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype   # bfloat16 when --bf16

    # Build a minimal batch (1 sample), cast floats to model dtype to match weights
    sample = train_dataset[0]
    batch  = default_collate([sample])
    batch  = {k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device))
              for k, v in batch.items() if isinstance(v, torch.Tensor)}

    q01 = model.action_q01.to(device, dtype=dtype)
    q99 = model.action_q99.to(device, dtype=dtype)
    anchors_phys_d = anchor_phys.to(device, dtype=dtype)
    anchors_norm_d = anchor_norm.to(device, dtype=dtype)

    # k+ selection
    k_plus = select_positive_anchor(anchors_phys_d, batch["action"], q01, q99)
    anchor_for_batch = anchors_norm_d[k_plus]

    # --- forward ---
    model.train()
    outputs = model(
        **batch,
        anchor=anchor_for_batch,
        sigma_anchor=0.0,   # warmup at step 0
        lambda_a=0.0,       # warmup at step 0
    )
    loss = outputs.loss
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"[smoke_test] forward loss = {loss.item():.6f}  ✓")

    # --- backward ---
    loss.backward()
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_grad_norm += p.grad.detach().float().norm() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    assert torch.isfinite(torch.tensor(total_grad_norm)), "grad_norm is not finite"
    print(f"[smoke_test] grad_norm   = {total_grad_norm:.6f}  ✓")

    # --- vlm grad check ---
    vlm_grad_count = sum(1 for p in model.vlm.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
    assert vlm_grad_count == 0, f"vlm.* has non-zero grads! ({vlm_grad_count} params)"
    print(f"[smoke_test] vlm.* grads = 0  ✓")

    # --- trainable count ---
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke_test] trainable params = {trainable:,}")

    # --- vlm hash ---
    vlm_hash = _compute_vlm_hash(model)
    print(f"[smoke_test] vlm hash = {vlm_hash[:32]}...")

    # --- mixture_logit shape ---
    if outputs.mixture_logit is not None:
        assert outputs.mixture_logit.shape == (batch["action"].shape[0],), \
            f"mixture_logit shape mismatch: {outputs.mixture_logit.shape}"
        print(f"[smoke_test] mixture_logit shape = {tuple(outputs.mixture_logit.shape)}  ✓")
    else:
        print("[smoke_test] mixture_logit = None (expected when anchor is not None — check model)")

    print("\n[smoke_test] ALL CHECKS PASSED ✓")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsStage2A))
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

    # --- Model ---
    model = load_model_stage2a(model_args, pi0_config, training_args)

    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    # --- Tokenizer ---
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # --- Anchor buffers ---
    anchor_phys, anchor_norm = inject_anchor_buffers(model, training_args.anchor_cache_path)

    # --- Dataset ---
    train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)

    # --- Smoke test early exit ---
    if training_args.smoke_test:
        # Move model to GPU for smoke test
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        run_smoke_test(model, train_dataset, anchor_phys, anchor_norm, training_args)
        return

    # --- Warmup scheduler ---
    warmup = WarmupScheduler(
        warmup_steps=training_args.anchor_warmup_steps,
        sigma_anchor_max=training_args.sigma_anchor_max,
    )
    print(f"[Stage2A] anchor warmup: lambda_a + sigma_anchor over {training_args.anchor_warmup_steps} steps")
    print(f"[Stage2A] bce_strategy = {training_args.bce_strategy}")

    # --- Callbacks ---
    callbacks = [VLMHashCallback()]

    # --- Trainer ---
    trainer = GRPOStage2ATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        warmup_scheduler=warmup,
        anchor_centers_phys=anchor_phys,
        anchor_centers_norm=anchor_norm,
        bce_strategy=training_args.bce_strategy,
        bce_n_neg=training_args.bce_n_neg,
        bce_loss_weight=training_args.bce_loss_weight,
        bce_chunk_size=training_args.bce_chunk_size,
    )

    # --- Train ---
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()


if __name__ == "__main__":
    train()
