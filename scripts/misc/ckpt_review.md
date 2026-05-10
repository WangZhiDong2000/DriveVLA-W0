# Ckpt Review (Task 0.1)

- **Ckpt dir**: `/home/wang/Project/DriveVLA-W0/pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2`
- **Shards**: 4 files, total params: 9,073,333,251 (9.07B), total bytes (params only): 16.90 GB
- **Total tensor keys**: 596

## 1. Prefix histogram (top-level)

| prefix | num_keys | total numel | total bytes | dtype 分布 |
|---|---:|---:|---:|---|
| `action_decoder.*` | 4 | 2,102,275 (2.1M) | 4.01 MB | BF16:4 |
| `action_expert.*` | 290 | 574,995,456 (575.0M) | 1.07 GB | BF16:290 |
| `action_projector.*` | 7 | 3,160,064 (3.2M) | 6.03 MB | BF16:7 |
| `state_projector.*` | 4 | 1,063,936 (1.1M) | 2.03 MB | BF16:4 |
| `vlm.*` | 291 | 8,492,011,520 (8.49B) | 15.82 GB | BF16:291 |

✅ All top-level prefixes are within the expected set.

- `action_expert.layers.{N}.*` 实测层号集合: 32 layers, min=0, max=31
- `vlm.lm_head.*` 键: 1 个 → 存在 ✅

### `action_expert.*` 内部分桶（vs plan §0.2 #2 估计 150–250M）

- `embed_tokens.weight`: 189,052,928 (189.1M)
- `layers.*` 32 层合计: 385,941,504 (385.9M)
- `norm.weight`: 1,024 (1.0k)
- **合计**: 574,995,456 (575.0M)
- transformer-only（不含 embed_tokens）: 385,942,528 (385.9M)

**Sample attention shapes (layer 0)**:
  - `self_attn.k_proj.weight` shape=(1024, 1024)
  - `self_attn.o_proj.weight` shape=(1024, 4096)
  - `self_attn.q_proj.weight` shape=(4096, 1024)
  - `self_attn.v_proj.weight` shape=(1024, 1024)
**Sample MLP shapes (layer 0)**:
  - `mlp.down_proj.weight` shape=(1024, 512)
  - `mlp.gate_proj.weight` shape=(512, 1024)
  - `mlp.up_proj.weight` shape=(512, 1024)

⚠️ action_expert 总参数量 575.0M **超出** plan §0.2 #2 的 [150, 250]M 估计 — 主要由 `embed_tokens` (189.1M) 与 layer 内 32 query heads × head_dim=128（q/o_proj 是 1024↔4096）共同贡献。 plan v2.2 的估计未计入这两块。Task 0.2 阈值需相应放宽至覆盖实测值。

### Per-shard key count
- `model-00001-of-00004.safetensors`: 70 keys
- `model-00002-of-00004.safetensors`: 106 keys
- `model-00003-of-00004.safetensors`: 100 keys
- `model-00004-of-00004.safetensors`: 320 keys

## 2. trainer_state.json 摘要

| field | value |
|---|---|
| `global_step` | `10000` |
| `epoch` | `9.775171065493646` |
| `max_steps` | `10000` |
| `num_train_epochs` | `10` |
| `train_batch_size` | `12` |
| `save_steps` | `5000` |
| `eval_steps` | `400` |
| `logging_steps` | `10` |
| `best_metric` | `None` |
| `best_model_checkpoint` | `None` |
| `final_train_loss` | `0.034265828418731686` |
| `final_eval_loss` | `0.030875079333782196` |
| `last_step_loss` | `0.0179` |
| `last_step_grad_norm` | `0.3511170446872711` |
| `train_runtime_sec` | `176634.3361` |
| `train_samples_per_second` | `5.435` |
| `estimated_samples_seen` | `960007` |

### Tail step trend (last 10 logged steps)

| step | loss | grad_norm | lr |
|---:|---:|---:|---:|
| 9910 | 0.0224 | 0.27195653319358826 | 1.00989111991682e-06 |
| 9920 | 0.0205 | 1.5470420122146606 | 1.007815316235018e-06 |
| 9930 | 0.0161 | 0.5387211441993713 | 1.0059836760570947e-06 |
| 9940 | 0.0095 | 0.0813731700181961 | 1.0043962176427436e-06 |
| 9950 | 0.0116 | 0.23792590200901031 | 1.0030529568173957e-06 |
| 9960 | 0.0181 | 0.5694689750671387 | 1.0019539069720683e-06 |
| 9970 | 0.0312 | 1.5745723247528076 | 1.0010990790632335e-06 |
| 9980 | 0.021 | 0.7169158458709717 | 1.0004884816126956e-06 |
| 9990 | 0.012 | 0.43844276666641235 | 1.0001221207075302e-06 |
| 10000 | 0.0179 | 0.3511170446872711 | 1e-06 |

## 3. training_args.bin 摘要

| field | value |
|---|---|
| `action_loss_weight` | `1.0` |
| `action_sample_steps` | `10` |
| `bf16` | `True` |
| `eval_steps` | `400` |
| `evaluation_strategy` | `steps` |
| `fp16` | `False` |
| `freeze_vlm` | `False` |
| `gradient_accumulation_steps` | `1` |
| `learning_rate` | `5e-05` |
| `lr_scheduler_type` | `cosine_with_min_lr` |
| `max_steps` | `10000` |
| `num_train_epochs` | `3.0` |
| `output_dir` | `logs/train_navsim_pi0_vava_from_nuplan_ft` |
| `per_device_train_batch_size` | `12` |
| `run_name` | `logs/train_navsim_pi0_vava_from_nuplan_ft` |
| `save_steps` | `5000` |
| `train_action_only` | `False` |
| `warmup_ratio` | `0.0` |
| `warmup_steps` | `50` |
| `weight_decay` | `0.1` |

## 4. config.json 关键字段

```json
{
  "model_type": "Emu3Pi0",
  "architectures": [
    "Emu3Pi0"
  ],
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "max_position_embeddings": 1400,
  "vocab_size": 184622,
  "torch_dtype": "bfloat16",
  "action_dim": 3,
  "action_frames": 8,
  "action_loss_weight": 1.0,
  "action_sample_steps": 10,
  "vision_loss_weight": 0.5,
  "vision_token_weight": 0.5,
  "vlm_loss_weight": 1.0,
  "freeze_vlm": false,
  "train_action_only": false,
  "action_config": {
    "attention_dropout": 0.0,
    "head_dim": 128,
    "hidden_size": 1024,
    "intermediate_size": 512,
    "max_position_embeddings": 2400,
    "model_type": "Emu3",
    "pre_action_frames": 3,
    "torch_dtype": "bfloat16"
  },
  "vlm_config": {
    "action_dim": 3,
    "architectures": [
      "Emu3MoE"
    ],
    "image_area": 262144,
    "max_position_embeddings": 1400,
    "model_type": "Emu3",
    "torch_dtype": "bfloat16",
    "vision_loss_weight": 0.5
  }
}
```

## 5. Pass / Fail 结论 (Task 0.1)

**PASS**:
- ✅ 顶层 prefix 集合 ⊆ 预期集合
- ✅ action_expert 层号 0..31 完整 (32 层)
- ✅ vlm.lm_head.* 存在（AR-WM 信号头）
- ✅ trainer_state: global_step=10000, epoch=9.78 ≈ 完整训练画像
- ✅ final eval_loss=0.0309

**FAIL**:
- (none)

### 一句话结论

> 该 ckpt 在 `logs/train_navsim_pi0_vava_from_nuplan_ft` 上训练（freeze_vlm=False, lr=5e-05），共 10000 steps (~9.78 epochs, train_batch_size=12, 估计 ~960007 samples 看过)，末次 eval_loss=0.0309。 训练画像与 plan v2.2 §0.2 #8 的「完整 navtrain」一致 → Phase 3 走 **adapt 模式（2k-4k steps）**；`action_expert` 实测 575M（含 embed_tokens 189M），plan §0.2 #2 的 150-250M 估计需在 Task 0.2 阈值与下游文档中校正为实测值。

---

## 6. from_pretrained 加载 + 参数量分组（Task 0.2）

加载形态对齐 [utils/train_pi0.py:149-163](../../utils/train_pi0.py#L149-L163) 的 `init_fresh_expert=False` 分支：`Emu3Pi0.from_pretrained(..., torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)`。**`pretrain_vlm_path=None`** 替换 train_pi0 默认的 `pretrain_vlm_path=ckpt_dir`，以跳过 `__init__` 内一次冗余的 `Emu3MoE.from_pretrained`（其在 Pi0 ckpt 上因 `vlm.*` 前缀错配而几乎全部 missing），把 CPU 峰值内存降到 ~17GB。外层 `from_pretrained` 完整加载所有权重 — 验证 missing/unexpected 全 0。

- **`missing_keys`**: 0 (first 5: `[]`)
- **`unexpected_keys`**: 0 (first 5: `[]`)
- **`mismatched_keys`**: `[]`
- ✅ Loading clean (Task 0.2 阶段尚未引入新 anchor/mixture heads)

### Submodule 参数量

| submodule | numel |
|---|---:|
| `vlm.*` | 8,492,011,520 (8.49B) |
| `vlm.lm_head.*` | 756,211,712 (756.2M) |
| `action_expert.*` | 574,995,456 (575.0M) |
| `state_projector.*` | 1,063,936 (1.1M) |
| `action_projector.*` | 3,160,064 (3.2M) |
| `action_decoder.*` | 2,102,275 (2.1M) |
| `tau_emb.*` | 0 |
| **TOTAL (model.parameters())** | 9,073,333,251 (9.07B) |

✅ `action_expert` numel = 574,995,456 (575.0M) 在 [500,000,000 (500.0M), 700,000,000 (700.0M)] 实测调整后区间内。

**Plan §0.2 #2 校正**：原估 150–250M 未计入 `action_expert.embed_tokens` (184622 × 1024 = 189M) 与 GQA q/o_proj 的 32 query heads (1024↔4096 投影)；实测 575M 是这两块共同贡献。后续文档/阈值需引用本节实测值。

### Top-level named_parameters prefix

| prefix | numel | tensors |
|---|---:|---:|
| `action_decoder.*` | 2,102,275 (2.1M) | 4 |
| `action_expert.*` | 574,995,456 (575.0M) | 290 |
| `action_projector.*` | 3,160,064 (3.2M) | 7 |
| `state_projector.*` | 1,063,936 (1.1M) | 4 |
| `vlm.*` | 8,492,011,520 (8.49B) | 291 |

✅ 所有 top-level prefix 都在预期集合内。

### Trainable / Frozen 分组（plan §Task 0.2 + §6 超参表）

Freeze 策略：`vlm.*`（含 `lm_head`）+ `state_projector.*` → `requires_grad=False`
Trainable: `action_expert.*` / `action_projector.*` / `action_decoder.*` / `tau_emb.*`

- **Trainable 总量**: 580,257,795 (580.3M)
- **Frozen 总量**: 8,493,075,456 (8.49B)
- **Total**: 9,073,333,251 (9.07B)

| group | prefix | numel |
|---|---|---:|
| trainable | `action_decoder.*` | 2,102,275 (2.1M) |
| trainable | `action_expert.*` | 574,995,456 (575.0M) |
| trainable | `action_projector.*` | 3,160,064 (3.2M) |
| frozen | `state_projector.*` | 1,063,936 (1.1M) |
| frozen | `vlm.*` | 8,492,011,520 (8.49B) |

## 7. vlm 参数 hash（Phase 3 Task 5.2 anchor）

以 `sha256(name || dtype || shape || raw_bytes)` 顺序累加：

- `vlm_hash_t0` = `8a1c664ebcdc23b4b6ddc70ce34e1405bfc3f5f18f433ff54c4643670e13cac1`

Phase 3 训练 N steps 后比对此 hash；若 vlm.* 真正 frozen 则两值应严格相等。

## 8. Pass / Fail 结论 (Task 0.2)

**PASS**:
- ✅ from_pretrained loading clean
- ✅ action_expert numel ∈ 调整后区间
- ✅ top-level prefixes 落在预期集合
- ✅ freeze_vlm + state_projector frozen → trainable=580,257,795 (580.3M), frozen=8,493,075,456 (8.49B)
- ✅ vlm_hash_t0 落盘

**FAIL**:
- (none)
