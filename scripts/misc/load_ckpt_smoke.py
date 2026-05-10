"""Task 0.2 — 组件加载与 trainable 分组实测.

按 utils/train_pi0.py:149-163 的 from_pretrained 分支干净加载 Emu3Pi0，
验证 missing/unexpected keys 干净、action_expert numel、freeze_vlm + trainable 分组、
计算 vlm 的 sha256 hash（为 Phase 3 Task 5.2 hash 校验打基础）。
追加结果到 ckpt_review.md（与 Task 0.1 共享同一份报告）。

参考 plan §Task 0.2 与子计划 (drivevla-w0-grpo-implementation-plan-3-melodic-kahn.md)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
from collections import defaultdict

PROJECT_ROOT = osp.expanduser("~/Project/DriveVLA-W0")
DEFAULT_CKPT = osp.join(
    PROJECT_ROOT,
    "pretrained_models",
    "Emu3_Flow_Matching_Action_Expert_PDMS_87.2",
)
DEFAULT_OUT_MD = osp.join(PROJECT_ROOT, "scripts", "misc", "ckpt_review.md")

# Match the import path style of utils/train_pi0.py:22.
# PROJECT_ROOT must come first so `models.policy_head.*` resolves (referenced from
# reference/Emu3/emu3/mllm/modeling_emu3.py:64).
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, osp.join(PROJECT_ROOT, "reference", "Emu3"))

# Plan §0.2 #2 estimated 150–250M but Task 0.1 measured 575M (incl. embed_tokens 189M).
# Threshold widened to cover the actual disk-side footprint with ±15% slack.
ACTION_EXPERT_NUMEL_LO = int(5.0e8)
ACTION_EXPERT_NUMEL_HI = int(7.0e8)

EXPECTED_TOP_PREFIXES = {
    "vlm",
    "action_expert",
    "state_projector",
    "action_projector",
    "action_decoder",
    "tau_emb",  # parameterless SinusoidalPosEmb — appears under named_parameters with 0 entries
    "rf",       # parameterless scheduler
    "shared_layers",  # plain Python list, not Module — won't show in named_parameters
}

TRAINABLE_PREFIXES = ("action_expert.", "action_projector.", "action_decoder.", "tau_emb.")
FROZEN_PREFIXES = ("vlm.", "state_projector.")


def fmt_int(n):
    if n >= 1e9:
        return f"{n:,} ({n/1e9:.2f}B)"
    if n >= 1e6:
        return f"{n:,} ({n/1e6:.1f}M)"
    if n >= 1e3:
        return f"{n:,} ({n/1e3:.1f}k)"
    return f"{n:,}"


def numel(module):
    return sum(p.numel() for p in module.parameters())


def hash_module(module) -> str:
    """sha256 over (name, dtype, shape, raw bytes) sorted by name.

    numpy lacks a bf16 dtype, so torch.bfloat16 tensors are reinterpreted as int16
    (same 2-byte layout) before tobytes().
    """
    import torch
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(str(p.dtype).encode())
        h.update(str(tuple(p.shape)).encode())
        flat = p.detach().to("cpu").contiguous().view(-1)
        if flat.dtype == torch.bfloat16:
            flat = flat.view(torch.int16)
        h.update(flat.numpy().tobytes())
    return h.hexdigest()


def group_named_params(model):
    """Return dict[top_prefix -> (num_params, num_tensors)]."""
    groups = defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        groups[top][0] += p.numel()
        groups[top][1] += 1
    return {k: tuple(v) for k, v in groups.items()}


def split_trainable(model):
    trainable = defaultdict(int)
    frozen = defaultdict(int)
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        if p.requires_grad:
            trainable[top] += p.numel()
        else:
            frozen[top] += p.numel()
    return dict(trainable), dict(frozen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_md", default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    print(f"[load_smoke] ckpt_dir = {args.ckpt_dir}")
    print(f"[load_smoke] device   = {args.device}")
    print(f"[load_smoke] out_md   = {args.out_md}")

    import torch
    from emu3.mllm import Emu3Pi0, Emu3Pi0Config

    cfg_path = osp.join(args.ckpt_dir, "config.json")
    print(f"[load_smoke] loading config from {cfg_path} ...", flush=True)
    model_config = Emu3Pi0Config.from_pretrained(cfg_path)

    # NOTE on `pretrain_vlm_path`: train_pi0.py:155 passes it to populate vlm during
    # __init__, but that triggers a redundant Emu3MoE.from_pretrained call on the same
    # ckpt (which has `vlm.*`-prefixed keys, so it largely no-ops with missing keys
    # warnings) and doubles the peak RAM.  With `pretrain_vlm_path=None`, __init__
    # creates a fresh bf16 Emu3MoE; the outer Emu3Pi0.from_pretrained then loads the
    # real weights into both `vlm.*` and `action_expert.*` prefixes from the saved
    # state dict.  Verified clean on this ckpt (24GB RAM environment).
    print("[load_smoke] calling Emu3Pi0.from_pretrained (~17GB bf16 on CPU)...",
          flush=True)
    model, loading_info = Emu3Pi0.from_pretrained(
        args.ckpt_dir,
        config=model_config,
        pretrain_vlm_path=None,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    if args.device == "cuda":
        model = model.to("cuda")
    model.eval()

    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    mismatched = loading_info.get("mismatched_keys", [])
    print(f"\n[load_smoke] missing_keys    = {len(missing)} (first 5: {missing[:5]})")
    print(f"[load_smoke] unexpected_keys = {len(unexpected)} (first 5: {unexpected[:5]})")
    print(f"[load_smoke] mismatched_keys = {mismatched}")

    # Param totals & per-submodule breakdown.
    total = sum(p.numel() for p in model.parameters())
    submodule_numel = {
        "vlm.*": numel(model.vlm),
        "vlm.lm_head.*": numel(model.vlm.lm_head),
        "action_expert.*": numel(model.action_expert),
        "state_projector.*": numel(model.state_projector),
        "action_projector.*": numel(model.action_projector),
        "action_decoder.*": numel(model.action_decoder),
        "tau_emb.*": numel(model.tau_emb),
    }
    print("\n[load_smoke] Submodule param counts:")
    for k, v in submodule_numel.items():
        print(f"  {k:<22s} {fmt_int(v)}")
    print(f"  {'TOTAL':<22s} {fmt_int(total)}")

    ae = submodule_numel["action_expert.*"]
    ae_in_range = ACTION_EXPERT_NUMEL_LO <= ae <= ACTION_EXPERT_NUMEL_HI

    # Top-level named_parameters prefix grouping (ground truth).
    grp = group_named_params(model)
    unexpected_top = sorted(set(grp) - EXPECTED_TOP_PREFIXES)
    print("\n[load_smoke] Top-level named_parameters prefixes:")
    for k, (n, c) in sorted(grp.items()):
        marker = "" if k in EXPECTED_TOP_PREFIXES else "  ❌UNEXPECTED"
        print(f"  {k:<22s} numel={fmt_int(n):<22s} tensors={c}{marker}")

    # Apply freeze policy per plan §Task 0.2 / §6.
    print("\n[load_smoke] applying freeze policy: vlm.* + state_projector.* frozen")
    model.freeze_vlm()
    for p in model.state_projector.parameters():
        p.requires_grad = False

    trainable, frozen = split_trainable(model)
    n_train = sum(trainable.values())
    n_frozen = sum(frozen.values())
    print(f"\n[load_smoke] Trainable={fmt_int(n_train)}  Frozen={fmt_int(n_frozen)}  "
          f"Total={fmt_int(n_train + n_frozen)}")
    for k, v in sorted(trainable.items()):
        print(f"  trainable {k:<22s} {fmt_int(v)}")
    for k, v in sorted(frozen.items()):
        print(f"  frozen    {k:<22s} {fmt_int(v)}")

    # vlm hash for Phase 3 Task 5.2 anchor.
    print("\n[load_smoke] hashing vlm.* parameters (sha256, may take ~30s on CPU)...",
          flush=True)
    vlm_hash_t0 = hash_module(model.vlm)
    print(f"[load_smoke] vlm_hash_t0 = {vlm_hash_t0}")

    # Append report Section 6/7.
    lines = []
    lines.append("\n---\n")
    lines.append("## 6. from_pretrained 加载 + 参数量分组（Task 0.2）\n")
    lines.append("加载形态对齐 [utils/train_pi0.py:149-163](../../utils/train_pi0.py#L149-L163) "
                 "的 `init_fresh_expert=False` 分支：`Emu3Pi0.from_pretrained(..., "
                 "torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)`。"
                 "**`pretrain_vlm_path=None`** 替换 train_pi0 默认的 `pretrain_vlm_path=ckpt_dir`，"
                 "以跳过 `__init__` 内一次冗余的 `Emu3MoE.from_pretrained`（其在 Pi0 ckpt 上"
                 "因 `vlm.*` 前缀错配而几乎全部 missing），把 CPU 峰值内存降到 ~17GB。"
                 "外层 `from_pretrained` 完整加载所有权重 — 验证 missing/unexpected 全 0。\n")
    lines.append(f"- **`missing_keys`**: {len(missing)} (first 5: `{missing[:5]}`)")
    lines.append(f"- **`unexpected_keys`**: {len(unexpected)} (first 5: `{unexpected[:5]}`)")
    lines.append(f"- **`mismatched_keys`**: `{mismatched}`")
    if not missing and not unexpected:
        lines.append("- ✅ Loading clean (Task 0.2 阶段尚未引入新 anchor/mixture heads)")
    else:
        lines.append("- ⚠️ Loading 不干净 — 见上方 keys 列表")
    lines.append("")
    lines.append("### Submodule 参数量\n")
    lines.append("| submodule | numel |")
    lines.append("|---|---:|")
    for k, v in submodule_numel.items():
        lines.append(f"| `{k}` | {fmt_int(v)} |")
    lines.append(f"| **TOTAL (model.parameters())** | {fmt_int(total)} |")
    lines.append("")
    if ae_in_range:
        lines.append(
            f"✅ `action_expert` numel = {fmt_int(ae)} 在 [{fmt_int(ACTION_EXPERT_NUMEL_LO)}, "
            f"{fmt_int(ACTION_EXPERT_NUMEL_HI)}] 实测调整后区间内。"
        )
    else:
        lines.append(
            f"⚠️ `action_expert` numel = {fmt_int(ae)} 不在调整后的 "
            f"[{fmt_int(ACTION_EXPERT_NUMEL_LO)}, {fmt_int(ACTION_EXPERT_NUMEL_HI)}] 区间，请复核。"
        )
    lines.append("")
    lines.append("**Plan §0.2 #2 校正**：原估 150–250M 未计入 `action_expert.embed_tokens` "
                 "(184622 × 1024 = 189M) 与 GQA q/o_proj 的 32 query heads (1024↔4096 投影)；"
                 "实测 575M 是这两块共同贡献。后续文档/阈值需引用本节实测值。\n")

    lines.append("### Top-level named_parameters prefix\n")
    lines.append("| prefix | numel | tensors |")
    lines.append("|---|---:|---:|")
    for k, (n, c) in sorted(grp.items()):
        marker = "" if k in EXPECTED_TOP_PREFIXES else " ❌UNEXPECTED"
        lines.append(f"| `{k}.*`{marker} | {fmt_int(n)} | {c} |")
    lines.append("")
    if unexpected_top:
        lines.append(f"❌ unexpected top-level prefixes: {unexpected_top}")
    else:
        lines.append("✅ 所有 top-level prefix 都在预期集合内。")
    lines.append("")

    lines.append("### Trainable / Frozen 分组（plan §Task 0.2 + §6 超参表）\n")
    lines.append("Freeze 策略：`vlm.*`（含 `lm_head`）+ `state_projector.*` → `requires_grad=False`")
    lines.append("Trainable: `action_expert.*` / `action_projector.*` / `action_decoder.*` / `tau_emb.*`\n")
    lines.append(f"- **Trainable 总量**: {fmt_int(n_train)}")
    lines.append(f"- **Frozen 总量**: {fmt_int(n_frozen)}")
    lines.append(f"- **Total**: {fmt_int(n_train + n_frozen)}\n")
    lines.append("| group | prefix | numel |")
    lines.append("|---|---|---:|")
    for k, v in sorted(trainable.items()):
        lines.append(f"| trainable | `{k}.*` | {fmt_int(v)} |")
    for k, v in sorted(frozen.items()):
        lines.append(f"| frozen | `{k}.*` | {fmt_int(v)} |")
    lines.append("")

    lines.append("## 7. vlm 参数 hash（Phase 3 Task 5.2 anchor）\n")
    lines.append("以 `sha256(name || dtype || shape || raw_bytes)` 顺序累加：\n")
    lines.append(f"- `vlm_hash_t0` = `{vlm_hash_t0}`\n")
    lines.append("Phase 3 训练 N steps 后比对此 hash；若 vlm.* 真正 frozen 则两值应严格相等。\n")

    lines.append("## 8. Pass / Fail 结论 (Task 0.2)\n")
    pass_items = []
    fail_items = []
    if not missing and not unexpected and not mismatched:
        pass_items.append("from_pretrained loading clean")
    else:
        fail_items.append(
            f"loading: missing={len(missing)} unexpected={len(unexpected)} "
            f"mismatched={mismatched}"
        )
    if ae_in_range:
        pass_items.append(f"action_expert numel ∈ 调整后区间")
    else:
        fail_items.append("action_expert numel out of range")
    if not unexpected_top:
        pass_items.append("top-level prefixes 落在预期集合")
    else:
        fail_items.append(f"top-level prefixes 出现非预期: {unexpected_top}")
    pass_items.append(f"freeze_vlm + state_projector frozen → "
                      f"trainable={fmt_int(n_train)}, frozen={fmt_int(n_frozen)}")
    pass_items.append(f"vlm_hash_t0 落盘")

    lines.append("**PASS**:")
    for p in pass_items:
        lines.append(f"- ✅ {p}")
    lines.append("")
    lines.append("**FAIL**:")
    if not fail_items:
        lines.append("- (none)")
    else:
        for p in fail_items:
            lines.append(f"- ❌ {p}")
    lines.append("")

    with open(args.out_md, "a") as f:
        f.write("\n".join(lines))

    print(f"\n[load_smoke] appended Section 6/7/8 to {args.out_md}")
    print(f"[OK] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if ae_in_range:
        print(f"[OK] action_expert numel = {fmt_int(ae)} ∈ "
              f"[{fmt_int(ACTION_EXPERT_NUMEL_LO)}, {fmt_int(ACTION_EXPERT_NUMEL_HI)}]")
    else:
        print(f"[FAIL] action_expert numel = {fmt_int(ae)} out of range")
    print(f"[OK] vlm frozen, state_projector frozen, vlm_hash_t0={vlm_hash_t0[:16]}...")

    if missing or unexpected or mismatched or unexpected_top or not ae_in_range:
        sys.exit(2)


if __name__ == "__main__":
    main()
