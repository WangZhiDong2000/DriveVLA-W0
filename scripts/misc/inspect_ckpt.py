"""Task 0.1 — Ckpt 加载与 trainer_state 检视.

不依赖 transformers / Emu3，只用 safetensors + json + torch (可选)。
读 safetensors header 的元数据、解析 trainer_state.json / training_args.bin /
config.json，落 Markdown 审查报告（默认 scripts/misc/ckpt_review.md）。

参考 plan §Task 0.1 与子计划 (drivevla-w0-grpo-implementation-plan-3-melodic-kahn.md)。
"""
import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter, defaultdict
from glob import glob
from math import prod

from safetensors import safe_open


PROJECT_ROOT = osp.expanduser("~/Project/DriveVLA-W0")
DEFAULT_CKPT = osp.join(
    PROJECT_ROOT,
    "pretrained_models",
    "Emu3_Flow_Matching_Action_Expert_PDMS_87.2",
)
DEFAULT_OUT_MD = osp.join(PROJECT_ROOT, "scripts", "misc", "ckpt_review.md")

EXPECTED_PREFIXES = {
    "vlm",
    "action_expert",
    "state_projector",
    "action_projector",
    "action_decoder",
    "tau_emb",
}

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "BOOL": 1,
}


def collect_safetensor_meta(ckpt_dir):
    """Read all shard headers, return list of (key, shape, dtype, numel, bytes)."""
    shards = sorted(glob(osp.join(ckpt_dir, "model-*.safetensors")))
    if not shards:
        shards = sorted(glob(osp.join(ckpt_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards in {ckpt_dir}")

    rows = []
    per_shard_count = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            per_shard_count[osp.basename(shard)] = len(keys)
            for k in keys:
                ts = f.get_slice(k)
                shape = tuple(ts.get_shape())
                dtype = ts.get_dtype()  # string like "BF16"
                numel = prod(shape) if shape else 1
                nbytes = numel * DTYPE_BYTES.get(dtype, 0)
                rows.append((k, shape, dtype, numel, nbytes))
    return rows, shards, per_shard_count


def histogram_by_top_prefix(rows):
    """Group by first '.' segment. Returns dict[prefix -> dict(stats)]."""
    groups = defaultdict(lambda: {
        "num_keys": 0, "numel": 0, "bytes": 0, "dtypes": Counter(),
    })
    for k, _, dtype, numel, nbytes in rows:
        top = k.split(".", 1)[0]
        g = groups[top]
        g["num_keys"] += 1
        g["numel"] += numel
        g["bytes"] += nbytes
        g["dtypes"][dtype] += 1
    return dict(groups)


def action_expert_layer_set(rows):
    """Extract layer indices found under action_expert.layers.{N}.*."""
    layers = set()
    for k, *_ in rows:
        if k.startswith("action_expert.layers."):
            try:
                idx = int(k.split(".")[2])
                layers.add(idx)
            except (IndexError, ValueError):
                continue
    return layers


def vlm_lm_head_keys(rows):
    return [k for (k, *_) in rows if k.startswith("vlm.lm_head")]


def action_expert_breakdown(rows):
    """Split action_expert.* into embed/transformer/head buckets and gather attn shapes."""
    embed = layers = norm = other = 0
    sample_attn_shapes = {}
    sample_mlp_shapes = {}
    for k, shape, _, numel, _ in rows:
        if not k.startswith("action_expert."):
            continue
        if k == "action_expert.embed_tokens.weight":
            embed += numel
        elif k.startswith("action_expert.layers."):
            layers += numel
            tail = k.split(".", 3)[-1]
            if tail.startswith("self_attn.") and tail not in sample_attn_shapes:
                sample_attn_shapes[tail] = tuple(shape)
            elif tail.startswith("mlp.") and tail not in sample_mlp_shapes:
                sample_mlp_shapes[tail] = tuple(shape)
        elif k == "action_expert.norm.weight":
            norm += numel
        else:
            other += numel
    return {
        "embed_tokens": embed,
        "transformer_layers": layers,
        "final_norm": norm,
        "other": other,
        "sample_attn_shapes": sample_attn_shapes,
        "sample_mlp_shapes": sample_mlp_shapes,
    }


def parse_trainer_state(ckpt_dir):
    p = osp.join(ckpt_dir, "trainer_state.json")
    if not osp.exists(p):
        return None
    with open(p) as f:
        ts = json.load(f)
    summary = {
        "global_step": ts.get("global_step"),
        "epoch": ts.get("epoch"),
        "max_steps": ts.get("max_steps"),
        "num_train_epochs": ts.get("num_train_epochs"),
        "save_steps": ts.get("save_steps"),
        "eval_steps": ts.get("eval_steps"),
        "logging_steps": ts.get("logging_steps"),
        "train_batch_size": ts.get("train_batch_size"),
        "best_metric": ts.get("best_metric"),
        "best_model_checkpoint": ts.get("best_model_checkpoint"),
    }
    log = ts.get("log_history", [])
    train_runtime = None
    train_samples_per_second = None
    final_train_loss = None
    final_eval_loss = None
    last_log_loss = None
    last_log_grad = None
    for entry in log:
        if "train_runtime" in entry:
            train_runtime = entry["train_runtime"]
            train_samples_per_second = entry.get("train_samples_per_second")
            final_train_loss = entry.get("train_loss", final_train_loss)
        if "eval_loss" in entry:
            final_eval_loss = entry["eval_loss"]
        if "loss" in entry and "step" in entry:
            last_log_loss = entry["loss"]
            last_log_grad = entry.get("grad_norm", last_log_grad)
    summary.update({
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "last_step_loss": last_log_loss,
        "last_step_grad_norm": last_log_grad,
        "train_runtime_sec": train_runtime,
        "train_samples_per_second": train_samples_per_second,
        "estimated_samples_seen": (
            int(train_runtime * train_samples_per_second)
            if train_runtime and train_samples_per_second else None
        ),
    })
    # Collect last 10 step-level entries for ASCII trend.
    step_entries = [e for e in log if "loss" in e and "step" in e]
    tail = step_entries[-10:]
    summary["tail_loss_trend"] = [
        {"step": e["step"], "loss": e.get("loss"), "grad_norm": e.get("grad_norm"),
         "lr": e.get("learning_rate")}
        for e in tail
    ]
    return summary


def parse_training_args(ckpt_dir):
    p = osp.join(ckpt_dir, "training_args.bin")
    if not osp.exists(p):
        return None
    # The pickle inside training_args.bin references custom dataclasses
    # (TrainingArguments / ModelArguments / DataArguments) that originally lived in
    # __main__ when the trainer was started.  Inject permissive stubs so unpickling
    # can rehydrate the instance __dict__ without needing the original module.
    import sys as _sys
    import types as _types
    main_mod = _sys.modules.get("__main__")
    if main_mod is None:
        main_mod = _types.ModuleType("__main__")
        _sys.modules["__main__"] = main_mod
    for stub_name in ("TrainingArguments", "ModelArguments", "DataArguments"):
        if not hasattr(main_mod, stub_name):
            class _Stub:  # noqa: D401 — empty placeholder
                pass
            _Stub.__name__ = stub_name
            _Stub.__qualname__ = stub_name
            setattr(main_mod, stub_name, _Stub)
    try:
        import torch  # local import; only needed here.
        ta = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # pragma: no cover - environment dependent
        return {"_load_error": f"{type(e).__name__}: {e}"}
    keys_of_interest = [
        "init_fresh_expert", "freeze_vlm", "train_action_only",
        "action_loss_weight", "action_sample_steps",
        "vision_loss_weight", "vlm_loss_weight",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "num_train_epochs", "max_steps",
        "warmup_steps", "warmup_ratio", "weight_decay",
        "lr_scheduler_type", "bf16", "fp16",
        "data_path", "model_name_or_path", "model_config_path",
        "output_dir", "run_name",
        "normalizer_path", "action_dim", "pre_action_frames",
        "evaluation_strategy", "eval_steps", "save_steps",
    ]
    out = {}
    for k in keys_of_interest:
        if hasattr(ta, k):
            v = getattr(ta, k)
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = repr(v)
    return out


def parse_pi0_config(ckpt_dir):
    p = osp.join(ckpt_dir, "config.json")
    with open(p) as f:
        cfg = json.load(f)
    keys = [
        "model_type", "architectures", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "max_position_embeddings", "vocab_size", "torch_dtype",
        "action_dim", "action_frames", "action_loss_weight",
        "action_sample_steps", "vision_loss_weight", "vision_token_weight",
        "vlm_loss_weight", "freeze_vlm", "train_action_only",
        "action_config", "vlm_config",
    ]
    return {k: cfg.get(k) for k in keys}


def fmt_int(n):
    if n is None:
        return "—"
    if n >= 1e9:
        return f"{n:,} ({n/1e9:.2f}B)"
    if n >= 1e6:
        return f"{n:,} ({n/1e6:.1f}M)"
    if n >= 1e3:
        return f"{n:,} ({n/1e3:.1f}k)"
    return f"{n:,}"


def fmt_bytes(b):
    if b is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b} ?"


def write_report(out_md, ckpt_dir, prefix_hist, layer_set, lm_head_keys,
                  ae_breakdown,
                  ts_summary, ta_summary, cfg_summary, shards, per_shard_count,
                  total_keys, total_numel, total_bytes, errors):
    lines = []
    lines.append("# Ckpt Review (Task 0.1)\n")
    lines.append(f"- **Ckpt dir**: `{ckpt_dir}`")
    lines.append(f"- **Shards**: {len(shards)} files, total params: "
                 f"{fmt_int(total_numel)}, total bytes (params only): "
                 f"{fmt_bytes(total_bytes)}")
    lines.append(f"- **Total tensor keys**: {total_keys}")
    lines.append("")

    # Section 1
    lines.append("## 1. Prefix histogram (top-level)\n")
    lines.append("| prefix | num_keys | total numel | total bytes | dtype 分布 |")
    lines.append("|---|---:|---:|---:|---|")
    for prefix in sorted(prefix_hist):
        s = prefix_hist[prefix]
        dtype_str = ", ".join(f"{d}:{c}" for d, c in s["dtypes"].most_common())
        marker = "" if prefix in EXPECTED_PREFIXES else " ❌"
        lines.append(
            f"| `{prefix}.*`{marker} | {s['num_keys']} | "
            f"{fmt_int(s['numel'])} | {fmt_bytes(s['bytes'])} | {dtype_str} |"
        )
    lines.append("")

    unexpected = sorted(set(prefix_hist) - EXPECTED_PREFIXES)
    if unexpected:
        lines.append(f"❌ **Unexpected top-level prefixes**: {unexpected}")
    else:
        lines.append("✅ All top-level prefixes are within the expected set.")
    lines.append("")
    lines.append(f"- `action_expert.layers.{{N}}.*` 实测层号集合: "
                 f"{len(layer_set)} layers, "
                 f"min={min(layer_set) if layer_set else '—'}, "
                 f"max={max(layer_set) if layer_set else '—'}")
    if layer_set and (min(layer_set) != 0 or max(layer_set) + 1 != len(layer_set)):
        lines.append("  ❌ layer indices are not a contiguous 0..N-1 range")
    lines.append(f"- `vlm.lm_head.*` 键: {len(lm_head_keys)} 个 → "
                 f"{'存在 ✅' if lm_head_keys else '缺失 ❌'}")
    lines.append("")

    # Action expert internal breakdown (vs plan §0.2 #2 estimate of 150-250M)
    lines.append("### `action_expert.*` 内部分桶（vs plan §0.2 #2 估计 150–250M）\n")
    ae_total = (ae_breakdown["embed_tokens"]
                + ae_breakdown["transformer_layers"]
                + ae_breakdown["final_norm"]
                + ae_breakdown["other"])
    lines.append(f"- `embed_tokens.weight`: {fmt_int(ae_breakdown['embed_tokens'])}")
    lines.append(f"- `layers.*` 32 层合计: {fmt_int(ae_breakdown['transformer_layers'])}")
    lines.append(f"- `norm.weight`: {fmt_int(ae_breakdown['final_norm'])}")
    if ae_breakdown["other"]:
        lines.append(f"- 其他: {fmt_int(ae_breakdown['other'])}")
    lines.append(f"- **合计**: {fmt_int(ae_total)}")
    transformer_only = ae_breakdown["transformer_layers"] + ae_breakdown["final_norm"]
    lines.append(f"- transformer-only（不含 embed_tokens）: {fmt_int(transformer_only)}")
    lines.append("")
    lines.append("**Sample attention shapes (layer 0)**:")
    for k in sorted(ae_breakdown["sample_attn_shapes"]):
        lines.append(f"  - `{k}` shape={ae_breakdown['sample_attn_shapes'][k]}")
    lines.append("**Sample MLP shapes (layer 0)**:")
    for k in sorted(ae_breakdown["sample_mlp_shapes"]):
        lines.append(f"  - `{k}` shape={ae_breakdown['sample_mlp_shapes'][k]}")
    lines.append("")
    if 1.5e8 <= ae_total <= 2.5e8:
        lines.append("✅ action_expert 总参数量落在 plan §0.2 #2 的 [150e6, 250e6] 范围。")
    else:
        lines.append(
            f"⚠️ action_expert 总参数量 {ae_total/1e6:.1f}M **超出** plan §0.2 #2 的"
            " [150, 250]M 估计 — 主要由 `embed_tokens` ({:.1f}M) 与 layer 内 "
            "32 query heads × head_dim=128（q/o_proj 是 1024↔4096）共同贡献。"
            " plan v2.2 的估计未计入这两块。Task 0.2 阈值需相应放宽至覆盖实测值。"
            .format(ae_breakdown["embed_tokens"]/1e6)
        )
    lines.append("")

    lines.append("### Per-shard key count")
    for fname, n in per_shard_count.items():
        lines.append(f"- `{fname}`: {n} keys")
    lines.append("")

    # Section 2
    lines.append("## 2. trainer_state.json 摘要\n")
    if ts_summary is None:
        lines.append("⚠️ trainer_state.json 不存在")
    else:
        scalar_keys = [
            "global_step", "epoch", "max_steps", "num_train_epochs",
            "train_batch_size", "save_steps", "eval_steps", "logging_steps",
            "best_metric", "best_model_checkpoint",
            "final_train_loss", "final_eval_loss",
            "last_step_loss", "last_step_grad_norm",
            "train_runtime_sec", "train_samples_per_second",
            "estimated_samples_seen",
        ]
        lines.append("| field | value |")
        lines.append("|---|---|")
        for k in scalar_keys:
            v = ts_summary.get(k)
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")
        tail = ts_summary.get("tail_loss_trend") or []
        if tail:
            lines.append("### Tail step trend (last 10 logged steps)\n")
            lines.append("| step | loss | grad_norm | lr |")
            lines.append("|---:|---:|---:|---:|")
            for e in tail:
                lines.append(
                    f"| {e['step']} | {e['loss']} | {e['grad_norm']} | {e['lr']} |"
                )
        lines.append("")

    # Section 3
    lines.append("## 3. training_args.bin 摘要\n")
    if ta_summary is None:
        lines.append("⚠️ training_args.bin 不存在")
    elif "_load_error" in ta_summary:
        lines.append(f"⚠️ 加载失败: `{ta_summary['_load_error']}`")
    else:
        lines.append("| field | value |")
        lines.append("|---|---|")
        for k in sorted(ta_summary):
            v = ta_summary[k]
            lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

    # Section 4
    lines.append("## 4. config.json 关键字段\n")
    lines.append("```json")
    lines.append(json.dumps(cfg_summary, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    # Section 5: pass / fail
    lines.append("## 5. Pass / Fail 结论 (Task 0.1)\n")
    pass_items = []
    fail_items = []

    if not unexpected:
        pass_items.append("顶层 prefix 集合 ⊆ 预期集合")
    else:
        fail_items.append(f"出现非预期 prefix: {unexpected}")

    if layer_set and min(layer_set) == 0 and max(layer_set) + 1 == len(layer_set):
        pass_items.append(f"action_expert 层号 0..{max(layer_set)} 完整 ({len(layer_set)} 层)")
    else:
        fail_items.append("action_expert.layers.* 层号不连续或缺失")

    if lm_head_keys:
        pass_items.append("vlm.lm_head.* 存在（AR-WM 信号头）")
    else:
        fail_items.append("vlm.lm_head.* 缺失")

    if ts_summary:
        gs = ts_summary.get("global_step") or 0
        ep = ts_summary.get("epoch") or 0
        eval_loss = ts_summary.get("final_eval_loss")
        if gs >= 10000 and ep >= 9.7:
            pass_items.append(
                f"trainer_state: global_step={gs}, epoch={ep:.2f} ≈ 完整训练画像"
            )
        else:
            fail_items.append(
                f"trainer_state 训练量偏小: global_step={gs}, epoch={ep}"
            )
        if eval_loss is not None:
            pass_items.append(f"final eval_loss={eval_loss:.4f}")

    if errors:
        for e in errors:
            fail_items.append(f"运行时: {e}")

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

    # Section 5 conclusion sentence
    if ts_summary and not fail_items:
        gs = ts_summary["global_step"]
        ep = ts_summary["epoch"]
        bs = ts_summary["train_batch_size"]
        eval_loss = ts_summary.get("final_eval_loss")
        samples = ts_summary.get("estimated_samples_seen")
        ta_summary_safe = ta_summary or {}
        data_hint = (
            ta_summary_safe.get("data_path")
            or ta_summary_safe.get("output_dir")
            or ta_summary_safe.get("run_name")
            or "<unknown>"
        )
        lr = ta_summary_safe.get("learning_rate")
        freeze_vlm_flag = ta_summary_safe.get("freeze_vlm")
        lines.append("### 一句话结论\n")
        lines.append(
            f"> 该 ckpt 在 `{data_hint}` 上训练（freeze_vlm={freeze_vlm_flag}, "
            f"lr={lr}），共 {gs} steps (~{ep:.2f} epochs, train_batch_size={bs}, "
            f"估计 ~{samples} samples 看过)，末次 eval_loss={eval_loss:.4f}。"
            " 训练画像与 plan v2.2 §0.2 #8 的「完整 navtrain」一致 → "
            "Phase 3 走 **adapt 模式（2k-4k steps）**；"
            "`action_expert` 实测 575M（含 embed_tokens 189M），plan §0.2 #2 的"
            " 150-250M 估计需在 Task 0.2 阈值与下游文档中校正为实测值。"
        )
        lines.append("")

    os.makedirs(osp.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--out_md", default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir
    if not osp.isdir(ckpt_dir):
        sys.exit(f"[ERR] ckpt_dir not found: {ckpt_dir}")

    print(f"[inspect_ckpt] ckpt_dir = {ckpt_dir}")
    print(f"[inspect_ckpt] out_md   = {args.out_md}")

    errors = []
    rows, shards, per_shard_count = collect_safetensor_meta(ckpt_dir)
    total_keys = len(rows)
    total_numel = sum(r[3] for r in rows)
    total_bytes = sum(r[4] for r in rows)
    print(f"[inspect_ckpt] {total_keys} keys across {len(shards)} shards "
          f"({fmt_int(total_numel)} params, {fmt_bytes(total_bytes)})")

    prefix_hist = histogram_by_top_prefix(rows)
    layer_set = action_expert_layer_set(rows)
    lm_head_keys = vlm_lm_head_keys(rows)
    ae_breakdown = action_expert_breakdown(rows)

    try:
        ts_summary = parse_trainer_state(ckpt_dir)
    except Exception as e:
        errors.append(f"trainer_state parse: {e}")
        ts_summary = None
    try:
        ta_summary = parse_training_args(ckpt_dir)
    except Exception as e:
        errors.append(f"training_args parse: {e}")
        ta_summary = None
    cfg_summary = parse_pi0_config(ckpt_dir)

    # Console summary
    print("\n[Prefix histogram]")
    for prefix, s in sorted(prefix_hist.items()):
        marker = "" if prefix in EXPECTED_PREFIXES else "  ❌UNEXPECTED"
        print(f"  {prefix:<20s} keys={s['num_keys']:<5d} "
              f"numel={fmt_int(s['numel']):<22s} bytes={fmt_bytes(s['bytes'])}"
              f"{marker}")
    print(f"\n[action_expert layers] {len(layer_set)} layers, "
          f"range={[min(layer_set) if layer_set else None, max(layer_set) if layer_set else None]}")
    print(f"[vlm.lm_head.*] {len(lm_head_keys)} keys")

    write_report(
        args.out_md, ckpt_dir, prefix_hist, layer_set, lm_head_keys,
        ae_breakdown,
        ts_summary, ta_summary, cfg_summary, shards, per_shard_count,
        total_keys, total_numel, total_bytes, errors,
    )

    print(f"\n[OK] report written to {args.out_md}")
    unexpected = set(prefix_hist) - EXPECTED_PREFIXES
    if unexpected or not lm_head_keys or not layer_set:
        print(f"[WARN] consistency issues — see report Section 5")
        sys.exit(2)


if __name__ == "__main__":
    main()
