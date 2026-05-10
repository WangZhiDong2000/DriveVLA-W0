# Technical Report
## DriveVLA-W0 + Flow Matching GRPO: Model Architecture and Component Specification

> **Companion document to the implementation plan (v2.2)**
> Scope: NAVSIM-scale Phase 1 / Flow Matching Action Expert / GRPO via DiffusionDrive v2 RL stack
> **Backbone: Emu3 (8B) VLM + AR-WM signal head (= `vlm.lm_head` shifted CE on visual tokens)**
> Version: 2.2 — code-level reconciliation pass over v2.0
> 主要修订：实际 ckpt 结构（无独立 world_model 模块）、Action Expert 真实大小、time 约定与代码同向、SharedLayer 替换"对称 Joint Attention"、multiplicative 探索噪声、IL 项 = vs GT L1、Mode Selector 重设计、参数量更正

---

## Abstract

This report specifies the model architecture and component-level details of a hybrid system that integrates DiffusionDrive v2's Anchored GRPO reinforcement learning into DriveVLA-W0's `Emu3Pi0` MoE Vision-Language-Action stack. The Stage 1 foundation is the locally available HuggingFace-style ckpt at `pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2/`, which contains a single `Emu3Pi0` model with two top-level submodules: `vlm.*`（Emu3MoE 8B + lm_head, 即 visual-token AR World Model 信号头）and `action_expert.*`（Emu3Model with hidden=1024, ~150–250M params）. v2.2 在 v2.0 设计基础上做了**代码级修订**：取消 v2.0 中并不存在的独立 `world_model` 模块假设，把所谓 "AR World Model" 落实到 `vlm.lm_head` 上的 shifted CE；把 v2.0 反向的时间约定改回与 [models/policy_head/noise_schedulers.py](models/policy_head/noise_schedulers.py) 同向；把 additive 探索噪声改为 DD-v2 paper §3.1 推荐的 multiplicative 形式；把 v2.0 写成"unchanged inherit"的 Mode Selector 重设计为 VLM-conditioned 版本（本仓库无 BEV encoder）；把 Stage 2-B 的 IL 项从"vs frozen reference policy"改为 DD-v2 实际使用的"vs GT trajectory L1"。新增两个组件：Anchored FM 路径下的 anchor embedding（作为额外 prefix token 注入）+ Mixture Weight head（per-anchor score）。**所有 PDMS 阈值从绝对值（87.2/89.0/90.0）改为相对 Task 0.3 实测的 B0**，以应对起点 ckpt 真实状态不确定性（R11）。

---

## 1. Design Rationale

DriveVLA-W0 reports two paired backbone+WM configurations: VQ (Emu3 8B + AR-WM) and ViT (Qwen2.5-VL 7B + Diffusion-WM). At the time of this implementation, the public release contains the **VQ Stage 1 checkpoint** but not the ViT variant. This report therefore specifies the system on the VQ track. The selection consequences:

1. **Stage 1 checkpoint is paper-released**, removing a major implementation blocker. We consume it directly and never re-train the backbone or World Model from scratch.
2. **Action Expert remains Flow Matching** — the same algorithmic transfer target from DiffusionDrive v2 as before. The paper validates this combination at 70k frames (Table 4), where FM achieves the best ADE among the three expert variants.
3. **AR World Model serves only as auxiliary representation supervision** during pretraining (Stage 1, already done) and optionally in Stage 2-A. It is not used at inference, so its sequential-decoding cost is not a runtime concern.
4. **WM-as-RL-reward is deferred to Phase 2.** AR-WM's sequential decoding makes model-based RL reward more expensive than the Diffusion-WM alternative; for Phase 1, NAVSIM PDM scorer is the sole reward source.

The Anchored GRPO stack from DiffusionDrive v2 transfers cleanly because it operates in continuous trajectory space, decoupled from the backbone's discrete visual tokenization.

---

## 2. System Overview

### 2.1 Component inventory（v2.2 与实际代码对齐）

> **Codebase 真相**（[reference/Emu3/emu3/mllm/modeling_emu3.py:1926-2003](reference/Emu3/emu3/mllm/modeling_emu3.py#L1926-L2003)）：`Emu3Pi0` 仅有两大模块 `self.vlm`（`Emu3MoE`，含 `lm_head`）与 `self.action_expert`（`Emu3Model`）+ 4 个小头。"AR World Model" 在本仓库**不是**独立模块，而是 `vlm.lm_head` 上对未来视觉 token 的 shifted CE。

| Component（实际 prefix） | Source | Params | Trainable in Stage 2-B | Role |
|-----------|--------|--------|------------------------|------|
| `vlm.*`（Emu3 8B + lm_head） | DriveVLA-W0 release | ~8B | **完全 frozen**（含 lm_head） | 多模态 context encoder + AR-WM 信号源 |
| `action_expert.*`（Emu3Model，hidden=1024, intermediate=512, 32 layers） | DriveVLA-W0 release | **~150–250M（待实测）** | Yes, full | 对 action token 的 transformer，joint-attend vlm KV |
| `state_projector.*` | DriveVLA-W0 release | <1M | **frozen**（保留对 cmd 的现有理解） | (pre_action; cmd) → state token |
| `action_projector.*` / `action_decoder.*` / `tau_emb.*` | DriveVLA-W0 release | ~5M | Yes, full（与 expert 同步） | noisy action↔hidden / vector field 解码 / time emb |
| `anchor_embedding.*` | **New（v2.2）** | ~3M | Yes, full | Anchor (B, N_f, 3) → (B, action_hidden) 作为额外 prefix token |
| `mixture_weight_head.*` | **New（v2.2）** | ~2M | Yes, full | Per-anchor scalar logit |
| `mode_selector.*` | **New（v2.2，VLM-conditioned）** | ~10M | No (Stage 3 only) | 从 N_anchor·G 候选中选 top-1（重设计，非 DD-v2 BEV 版） |

Total trainable parameters in Stage 2-B：≈ **~150–250M（expert）+ ~5M（小头）+ ~5M（新 heads）**。Task 0.2 必须打印实测值并写入实验日志。

**v2.2 删除的 v2.0 表项**：
- ~~"LoRA only ~32M"~~：v2.2 默认 vlm 完全 frozen，不用 LoRA；R10 兜底的 LoRA r=8 只 ~8M（不是 32M），仅在 Phase 3 PDMS 卡死时启用。
- ~~"AR World Model ~250M, token head gradient-stopped"~~：本仓库无独立 WM 模块，"WM frozen" = `vlm.*` 全 frozen。

### 2.2 Information flow

Inputs (text command, image, past action, anchor set) are tokenized by Emu3's unified tokenizer family and feed into the VLA backbone. The backbone emits hidden-state features that simultaneously condition the AR World Model (training only) and the Joint Attention layer that bridges into the Action Expert. The Action Expert produces both the per-anchor trajectory through the Anchored FM head and per-anchor scores through the Mixture Weight head. The Mode Selector picks the final output τ* during inference.

---

## 3. Backbone and Tokenization

### 3.1 VLA backbone (inherited)

The backbone is **Emu3 8B**, a unified multimodal autoregressive transformer. All three modalities are tokenized into a single discrete vocabulary:

- Text $L_t$ — Emu3 BPE tokenizer
- Image $V_t$ — **MoVQGAN** tokenizer producing discrete visual tokens (vocab ≈ 32K)
- Past action $A_{t-1}$ — FAST tokenizer (Pertsch et al. 2025), Emu3 vocabulary extension

For Stage 2 (this work), the input sequence is **2VA** — current frame plus one past frame:

$$S_t = [L_{t-1},\ V_{t-1},\ A_{t-2},\ L_t,\ V_t,\ A_{t-1}]$$

(Stage 1 used 6VA. The 2VA reduction is for inference efficiency; the IL warmup phase 4k steps is dedicated to adapting the backbone to this shorter context.)

The backbone emits final-layer hidden states partitioned by modality:

$$\{F_L^t,\ F_V^t,\ F_A^t\} = \text{Emu3}(S_t)$$

Although the modalities are discrete in input, the hidden states $F^t$ are continuous transformer outputs and serve as conditioning features for both the World Model and the Action Expert.

### 3.2 Backbone freezing strategy（v2.2 修订）

The 8B backbone (`vlm.*`, including `lm_head`) is **完全 frozen** by default in Stage 2-A and 2-B. v2.0 文档中的 "LoRA r=16 ≈ 32M" 表述被删除，原因：
1. 实际 ckpt 已是 Stage 2 训过的状态，backbone 在 NAVSIM 2VA 上的表征已经稳定，再追加 LoRA 收益边际且增加风险。
2. 完全 frozen 让 Phase 5.2 的"hash 校验保护"零成本生效（forward 前后 vlm.* 参数 hash 必相等）。
3. R2（RL 梯度污染 WM 表征）随之自动消解。

**R10 兜底（默认不启用）**：仅当 Phase 3 navtest PDMS 卡在 < B0 − 2.0 持续 1k steps 时，启用 Tier-1 LoRA：
- Rank $r=8$, $\alpha=16$，仅在 `vlm.*.q_proj` 与 `vlm.*.v_proj` 上注入。
- 实际参数量 ≈ 32 layers × 4096 × (q+v) × r=8 × 2 ≈ **8M**（v2.0 文档的 "32M" 估算偏大 4×）。
- 启用后必须保存 LoRA delta 而非 merge，便于回滚和 reference policy 比较。

---

## 4. AR World Model (Stage 1 representation prior，Stage 2 不参与)

> **v2.2 关键澄清**：本仓库**没有独立的 `world_model.*` 模块**。所谓 "AR World Model" 实体上就是 `vlm.lm_head` 上对未来视觉 token 的 next-token CE。

**Stage 1（已完成，本工作不再重训）**：在 `Emu3MoE.forward`（[modeling_emu3.py:1439-1700](reference/Emu3/emu3/mllm/modeling_emu3.py#L1439-L1700)）中，对 6VA 序列做 visual-token 自回归 CE，权重项见 [modeling_emu3.py:2150-2206](reference/Emu3/emu3/mllm/modeling_emu3.py#L2150-L2206)：

$$\mathcal{L}_{WM\text{-}AR} = \text{CE}\left(\text{lm\_head}(F^t)_{\text{shift}},\ V^{t+1}_{\text{shift labels}}\right) \cdot \mathbf{1}[\text{token} \in \text{visual range}]\cdot w_{\text{vis}}$$

其中 visual range = `[bov_token_id, eov_token_id]`，w_vis = `vision_token_weight=0.5`。

**Stage 2-B treatment（v2.2 锁定）**：`vlm.*` 完全 frozen，**包括 `lm_head`**。Stage 2-B 的训练 forward 中：
- 不计算 `vlm_loss`、不调用 `lm_head`（避免无谓显存与算力消耗）
- 通过 `freeze_vlm()`（[utils/train_pi0.py:128-135 上下文](utils/train_pi0.py#L128)）锁定所有 vlm 参数 `requires_grad=False`
- Phase 5.2 加 hash 校验作 sanity check

v2.0 文档残留的 "Optionally add β·L_WM-AR auxiliary loss" 在 v2.2 删除——它与"完全 frozen + 不 forward" 互斥。如果未来要在 Stage 2-B 重新引入 WM 信号，应作为单独 ablation（Phase 7 future work）。

**推理路径**：与 Stage 1/2 训练一致，`lm_head` 不参与 action 推理路径——`sample_actions` ([modeling_emu3.py:2223-2336](reference/Emu3/emu3/mllm/modeling_emu3.py#L2223-L2336)) 只走 vlm hidden states → joint shared-layers → action_decoder。

---

## 5. MoE 与 SharedLayer（v2.2 与代码对齐）

> **v2.0 描述的"对称 Joint Attention"与实际代码不符**。本仓库使用 `Emu3Pi0SharedLayer` ([reference/Emu3/emu3/mllm/modeling_emu3.py:1726+](reference/Emu3/emu3/mllm/modeling_emu3.py#L1726)) 与定制 attention mask `create_causal_style_attention_mask` ([modeling_emu3.py:2521-2685](reference/Emu3/emu3/mllm/modeling_emu3.py#L2521-L2685))。v2.2 改写本节。

### 5.1 双 Expert 拓扑

- **VLA Expert (`self.vlm`)**：Emu3MoE 32 层，hidden=4096，head_dim=128，KV head=8（GQA）。
- **Action Expert (`self.action_expert`)**：Emu3Model 32 层，hidden=1024，intermediate=512，与 VLA Expert 严格 **同层数 L=32** 但维度更小。
- 每层封装在 `Emu3Pi0SharedLayer`（plain Python object，含 vlm_layer + action_layer 引用），便于 gradient checkpointing。

### 5.2 SharedLayer 实际行为（非对称耦合）

每层逐位置 forward：

```
input  : (vlm_h, action_h)  shapes (B, S_v, 4096), (B, S_a, 1024)
combined attention mask 4D : (B, 1, S_v + S_a, S_v + S_a)
   - VLM 内部：标准 causal mask
   - Action 内部：full attention（[modeling_emu3.py:2547](reference/Emu3/emu3/mllm/modeling_emu3.py#L2547)）
   - Action → VLM：第二个 BOA 之前的 VLM 位置可见
   - VLM → Action：默认屏蔽（VLM 不看 action token）
```

注意 Q/K/V 不是把 vlm_h 和 action_h 拼起来直接做单次注意力——两个 expert 各自有独立的 `q_proj/k_proj/v_proj`（不同 hidden 维度），分别投影到各自 head_dim，再以 mask 控制谁能看谁。这与 v2.0 文档说的 `Q=[Q_VLA;Q_AE], K=[K_VLA;K_AE]` 单次 attention 不一致——实现上更接近"action expert 用自己的 Q 同时 attend (action KV ⊕ vlm KV) 拼接，而 VLM 不变"。

**对 GRPO 改造的影响**：
1. 我们要在 action_h 前面增加一个 anchor token，因此 `action_seq_len` 从原来的 `1 + N_f` 增到 `2 + N_f`，`create_causal_style_attention_mask` 需要相应扩列（让 anchor token 与 state_token 同等可见性）。
2. R10 Tier-2 兜底"放开 action↔vlm 的 cross-attn 层"在本结构中具体指：解冻 `action_expert.layers[i].self_attn` 中 `k_proj/v_proj` 关于 vlm KV 的部分（实现上需要给 SharedLayer 加 hook 区分 action-self-attn 与 action-attend-vlm 两支）。

### 5.3 `state_token` 与新增 `anchor_token`（v2.2 设计点）

当前 action 端 input prefill：
```
state_token  = state_projector(concat[pre_action.flatten(), cmd_one_hot])  # (B, 1, 1024)
action_h0    = [state_token; action_projector(noisy_action, tau_emb)]      # (B, 1+N_f, 1024)
```

v2.2 注入 anchor 后：
```
anchor_token = anchor_embedding(a_k)         # (B, 1, 1024)
action_h0    = [state_token; anchor_token; action_projector(noisy_action, tau_emb)]  # (B, 2+N_f, 1024)
```

cmd 与 anchor 默认共存（保留 ckpt 已学到的 cmd 行为），R12 ablation 验证。

---

## 6. Anchored Flow Matching Head (v2.2 重写：与代码同向 + multiplicative 噪声)

> **重要**：本节时间约定与 [models/policy_head/noise_schedulers.py](models/policy_head/noise_schedulers.py) 的 `add_noise` 一致——**t=0 是 GT，t=1 是 noise / anchored 端点**，ODE 推理走 t=1→0。v2.0/v2.1 文档使用了反向约定，v2.2 修正。

### 6.1 Anchor library

K-Means 一次性离线构建 $N_{anchor}=20$ 个 anchor $\{a^k\}_{k=1}^{20}$（与 DD-v2 默认 `ego_fut_mode=20` 对齐；10/15/20 在 Plan §Phase 5.1 ablation）。聚类只在 (x,y) 上做（$\mathbb{R}^{N_f \times 2}$，N_f=8），heading 通道由 Bezier 重建或直接取 GT 值。每个 anchor 代表一种粗粒度驾驶意图。

### 6.2 Anchor embedding（v2.2 注入方式：额外 prefix token）

```
e^k = MLP_a(flatten(a^k)) ∈ R^{d_AE=1024}
```

注入位置：作为 action expert input 序列的**第二个 prefix token**，与 `state_token`（[pre_action; cmd]）并列：

```
action_h0 = [state_token; e^k; action_projector(x_t, tau_emb)]
```

不再像 v2.0 描述那样"在每层都拼接"——只在第一层 prefill 注入即可，后续 32 层 SharedLayer 自然把 anchor 信号传播到所有位置。Phase 3 用 $\lambda_a$ warmup（0→1, 500 steps）逐步释放 anchor 信号以避免冷启动退化（R9）。

### 6.3 Anchored straight-line path（训练时，与代码同向）

训练样本 $\tau_{GT} \in \mathbb{R}^{N_f \times 3}$，正样本 anchor 选择：

$$k^+ = \arg\min_k \|a^k_{(:,:2)} - \tau_{GT,(:,:2)}\|_2$$

**Anchored noise 端点**（multiplicative，2 标量广播；DD-v2 paper §3.1）：

$$x_1^{k^+} = a^{k^+} \odot (1 + \epsilon_{\text{mul}}),\quad \epsilon_{\text{mul}} = (\epsilon_{\text{long}},\ \epsilon_{\text{lat}}) \sim \mathcal{N}(0,\ \sigma_{\text{anchor}}^2 I_2)$$

仅在 (x,y) 通道乘 $(1 + \epsilon_{mul})$，heading 通道保持 $a^{k^+}$ 原值（避免 heading 量级被 multiplicative 误伤）。$\sigma_{anchor} = 0.04$ 默认。

**直线路径（与 [noise_schedulers.py](models/policy_head/noise_schedulers.py) `add_noise` 同向）**：

$$x_t^{k^+} = (1 - t) \cdot \tau_{GT} + t \cdot x_1^{k^+},\quad t \in [0, 1]$$

t=0 时 $x_t = \tau_{GT}$（数据端），t=1 时 $x_t = x_1^{k^+}$（noise 端）。

### 6.4 Vector field predictor（与现有 MSE 写法兼容）

Action Expert 末层经 `action_decoder` 解出 $v_\phi \in \mathbb{R}^{N_f \times 3}$：

$$v_\phi = \text{action\_decoder}(\text{SharedLayers}_{32}(x_t^k,\ t,\ e^k,\ F_{\text{vlm}}))$$

直线路径下目标向量场为常向量：

$$v^*(x_t^{k^+}, t) = x_1^{k^+} - \tau_{GT}$$

注意符号：与 v2.0 相比反向。这与现有代码 `F.mse_loss(noise - action, velo_t_pred)` ([modeling_emu3.py:2143](reference/Emu3/emu3/mllm/modeling_emu3.py#L2143)) 一致——`noise - action` 在 anchored 设定下退化为 `x_1 - τ_GT`。

IL training objective（**仅正 anchor 反传**）：

$$\mathcal{L}_{FM\text{-}IL} = \mathbb{E}_t\left[\ \|v_\phi(x_t^{k^+},\ t,\ e^{k^+},\ F_{\text{vlm}}) - (x_1^{k^+} - \tau_{GT})\|^2\ \right]$$

负 anchor 不施加 reconstruction supervision，复用 DD-v2 paper Eq.(4) 的 $y^k=1$ mask 逻辑——防止 collapse-to-mean。

### 6.5 ODE samplers（v2.2: multiplicative 探索噪声）

> **v2.0 的 additive Gaussian per-element step noise 被 DD-v2 paper §3.1 明文反对**（导致 jagged 轨迹）。v2.2 改为 multiplicative。

记 $\Delta t = -1/T$（与代码 `dt = -1/N_steps` 一致），ODE 走 t=1→0。

**训练采样器（stochastic，$T_{\text{trunc}}=10$ 步）**：

$$z_{t-\Delta t}^{\text{mean}} = z_t + \Delta t \cdot v_\phi(z_t, t, \ldots)$$

$$z_{t-\Delta t} = z_{t-\Delta t}^{\text{mean}} \odot (1 + \epsilon_{\text{step}}),\quad \epsilon_{\text{step}} = (\epsilon^{step}_{\text{long}},\ \epsilon^{step}_{\text{lat}}) \sim \mathcal{N}(0, \sigma_{\text{step}}^2 I_2),\ \sigma_{\text{step}} \geq 0.04$$

广播规则同 §6.3：仅 (x,y) 上乘 multiplicative，heading 不变。`(z_{t-Δt}, log_prob, z_mean)` 三元组返回，与 DD-v2 `DDIMScheduler_with_logprob` 接口一致。

**推理采样器（deterministic，$T_{\text{infer}}=2$ 步）**：

$$z_{t-\Delta t} = z_t + \Delta t \cdot v_\phi$$

由于 anchored 起点已在数据流形附近，2 步足够。

### 6.6 Policy log-likelihood（multiplicative 形式）

log π 写在 $\epsilon_{\text{step}}$ 上而不是 z 上：

$$\log \pi_\theta(\epsilon_{\text{step}}^{(t)}) = -\frac{1}{2 \sigma_{\text{logpi}}^2}\|\epsilon_{\text{step}}^{(t)}\|^2 + C,\quad \sigma_{\text{logpi}} \geq 0.10$$

`σ_logpi` 是写入 log π 的标准差（不同于 `σ_step` 的采样标准差，但默认数值相同除非要做 PPO-style ratio shaping）。min clip 0.10 与 DD-v2 一致（Risk R6）。

REINFORCE per-token loss：

$$\ell^{k,i,t} = -\exp\left(\log \pi_\theta(\epsilon_{\text{step}}^{(k,i,t)}) - \log \pi_\theta(\epsilon_{\text{step}}^{(k,i,t)}).\text{detach}()\right) \cdot A^{k,i}_{\text{trunc}}$$

按非零 token 平均（DD-v2 [model_rl.py:1099-1102](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1099-L1102)）。

---

## 7. Mixture Weight Head (New)

### 7.1 Purpose

The Mixture Weight head solves a problem that emerges when the Mode Selector (Stage 3) has not yet been trained: during Stage 2-B intermediate evaluation, the system needs *some* way to pick which anchor's trajectory to commit to. A small per-anchor scoring head fills this role and also provides classification supervision that helps shape the Anchor Embedding.

### 7.2 Structure

For each anchor $k$, after the Joint Attention computation, the anchor-conditioned Action Expert features $F_{\text{AE}}^k$ are pooled and passed through a 3-layer MLP:

$$\hat s^k = \text{MLP}_s(\text{Pool}(F_{\text{AE}}^k)) \in \mathbb{R}$$

Final anchor probabilities are obtained by softmax across anchors.

### 7.3 Training objective

In Stage 2-A (IL warm-up), the Mixture Weight head is trained with a per-anchor BCE against the positive-mode indicator:

$$\mathcal{L}_{BCE} = \sum_{k=1}^{N_{\text{anchor}}} \text{BCE}(\hat s^k,\ \mathbf{1}[k = k^+])$$

The total Stage 2-A loss becomes:

$$\mathcal{L}_{IL} = \mathcal{L}_{FM\text{-}IL} + \mathcal{L}_{BCE}$$

(No coefficient between the two — they are on similar scales and DiffusionDrive v2 uses unit weights.)

In Stage 2-B (GRPO), the Mixture Weight head continues to be trained via the IL regularization term but does not appear in the RL loss (the Mode Selector subsumes its role in the final inference pipeline).

---

## 8. Mode Selector（v2.2 重设计：VLM-conditioned，无 BEV）

> **v2.0/v2.1 写"structurally unchanged from DD-v2"是错的**。DD-v2 selector 用 BEV feature + map/agent queries，本仓库**没有 BEV encoder**。v2.2 改为基于 VLM hidden states 的轻量 selector。

### 8.1 输入与结构

输入：
- 候选 trajectories $\{\hat\tau^k\}_{k=1}^{K}$，K = N_anchor × G = 20 × 8 = 160（推理时由 anchored ODE 生成）
- VLM hidden states $F_V \in \mathbb{R}^{B \times S_v \times 4096}$（已 frozen）

结构（共 ~10M 参数）：

```
1. Trajectory encoder：MLP, R^{N_f × 3} → R^{d_sel=256}
   q_traj ∈ R^{B × K × 256}

2. Cross-Attention(q=q_traj, K=V=Linear(F_V) → R^{B × S_v × 256})
   单层多头（heads=4），把场景 context 注入每个 candidate

3. Self-Attention over candidates：让 K 个 candidates 互相比较
   单层多头（heads=4）

4. Score MLP：R^{256} → R
   ŝ^k ∈ R^{B × K}
```

### 8.2 训练目标

```
L_selector = BCE(ŝ^k, y^k) + λ_rank · MarginRank(top-1, others)
```
其中 $y^k = \mathbf{1}[\arg\max_k r^k]$（$r^k$ 是 PDM score）。Margin-Rank loss 与 DD-v2 同形：

$$L_{\text{rank}} = \sum_{k \neq k^*} \max(0,\ \text{margin} - (\hat s^{k^*} - \hat s^k))$$

### 8.3 训练数据与流程

Stage 3 独立训练 20 epochs，所有上游组件 frozen：
- 先用 Stage 2-B 最终 ckpt 在 navtrain 上 rollout K=160 条/scene + PDM score（缓存约 15GB）
- 数据增广：multiplicative noise std ∈ [0.1, 0.2] 加到 trajectory 上
- 1% 候选从 GTRS 固定词表采样（如可用，否则可省）

### 8.4 与 DD-v2 selector 的差异表

| 项 | DD-v2 selector | v2.2 selector |
|---|---|---|
| 场景 context 来源 | BEV feature + map/agent queries | VLM hidden states (frozen) |
| 空间 attention | Deformable cross-attn | 标准 cross-attn over VLM tokens |
| Coarse-to-Fine | 两阶段 | 单阶段（K=160 已足够小） |
| Loss | BCE + Margin-Rank | 同 |
| 参数量 | ~50M | ~10M |

---

## 9. Training Pipeline

### 9.1 Stage 1 — pretrained, inherited

Inherited from DriveVLA-W0. The 8B Emu3 backbone + AR World Model is jointly pretrained on 6VA sequences with:

$$\mathcal{L}_{\text{Stage 1}} = \mathcal{L}_{\text{Action}} + \beta \cdot \mathcal{L}_{WM\text{-}AR}$$

This work *consumes* the resulting checkpoint; **no Stage 1 training is performed**. The checkpoint is downloaded from the DriveVLA-W0 public release.

### 9.2 Stage 2-A — IL anchor 适配（2k 起步，可延至 4k）

**Inputs**: 起点 ckpt（`pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2/`，**不走 `init_fresh_expert=True`**），anchor library $\{a^k\}$
**Trainable**: `action_expert.*` + `action_projector.*` / `action_decoder.*` / `tau_emb.*` + `anchor_embedding.*` + `mixture_weight_head.*`
**Frozen**: `vlm.*`（含 lm_head）+ `state_projector.*`
**Objective**:

$$\mathcal{L}_{\text{Stage 2-A}} = \mathcal{L}_{FM\text{-}IL}(k^+\text{ only}) + \mathcal{L}_{BCE}$$

本阶段定位（v2.2 修订）：
1. 把 anchor 结构（embedding + mixture head）适配到已训过的 FM expert 上。
2. 让 anchored 路径（§6.3）替代纯 noise 路径，loss 公式不变（仍是 MSE on `target = x_1 - τ_GT`）。
3. **不再产出"frozen reference policy"**——v2.2 的 Stage 2-B IL 项是 vs GT L1，不需要 reference policy。如未来需做 KL ablation（Plan Task 7.5），再单独保存一份 frozen 副本。

**Pass criterion**: navtest PDMS ≥ B0 − 1.2（B0 = Task 0.3 实测的起点 ckpt baseline）。

### 9.3 Stage 2-B — GRPO main training (10 epochs)

**Inputs**: Stage 2-A 最终 ckpt 作初始化（**不再有 frozen reference**）
**Trainable**: 同 Stage 2-A
**Per-step procedure**（与 DD-v2 [model_rl.py forward_train_rl + get_rlloss](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 二段式对齐）：

**Pass 1（no_grad，rollout）**：
1. VLM forward（一次，frozen）→ KV cache
2. 对每个 anchor $k$：anchor_embedding($a^k$) 与 state_token 拼前缀
3. 起点 $x_1^{k,i} = a^k \odot (1 + \epsilon_{mul}^{k,i})$，每个 anchor 采 $G=8$ 个 $\epsilon_{mul}^{k,i}$
4. Stochastic ODE 走 $T_{trunc}=10$ 步 → trajectory $\hat\tau^{k,i}$，每步缓存 $(\epsilon_{step}, \log \pi_\theta)$
5. PDM scorer 并行评估 16 workers → reward $r^{k,i}$
6. **Intra-anchor advantage**（mean/std 在 G 维，与 DD-v2 [model_rl.py:886-889](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L886-L889) 对齐）：
   $$A^{k,i} = (r^{k,i} - \mu_k) / (\sigma_k + 10^{-4})$$
7. **Inter-anchor truncation**：碰撞 / drivable_area 违规 → $A_{trunc}^{k,i} = -1$；其他负值 → 0；并仅保留 $r^{k,i} > r_{GT} - 10^{-6}$ 的样本（DD-v2 [model_rl.py:892](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L892)）
8. 时序折扣：$A_{trunc}^{k,i,t} \leftarrow A_{trunc}^{k,i} \cdot \gamma^{T-t-1}$，$\gamma=0.8$

**Pass 2（with_grad，loss）**：
1. 用 Pass 1 缓存的 $x_t^{k,i}$ 重跑 forward 得到 $\log \pi_\theta^{\text{new}}$（VLM forward 仍用同一 KV cache，不重算）
2. **REINFORCE per-token loss**（DD-v2 [model_rl.py:1096](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1096)）：
   $$\ell^{k,i,t} = -\exp(\log \pi_\theta^{\text{new}} - \log \pi_\theta^{\text{new}}.\text{detach}()) \cdot A_{trunc}^{k,i,t}$$
3. RL loss 按非零 token 平均
4. **IL loss = vs GT L1**（DD-v2 [model_rl.py:1105-1112](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1105-L1112)）：
   $$\mathcal{L}_{IL} = \frac{1}{D_{\text{steps}} \cdot D_{\text{layers}}} \sum_{\text{step},\text{layer}} \frac{1}{B \cdot K \cdot N_f} \sum \|\hat\tau^{k,i}_{(:,:2)} - \tau_{GT,(:,:2)}\|_1$$
5. **自适应权重**：$\lambda_{IL} = 0.1$ if `has_positive` else $1.0$（DD-v2 [model_rl.py:1115-1117](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1115-L1117)）
6. Total: $\mathcal{L} = \mathcal{L}_{RL} + \lambda_{IL} \cdot \mathcal{L}_{IL}$

**Hyperparameters**: $\gamma = 0.8$, batch size 512, AdamW with cosine schedule, bf16, gradient clip norm 1.0, $\sigma_{step} \geq 0.04$, $\sigma_{logpi} \geq 0.10$。
**Pass criterion**: navtest PDMS ≥ B0 + 2.0（复制 DD-v2 IL→GRPO 的 ~+3 PDMS 改善幅度，留 1 PDMS 给 Mode Selector 后续）。

### 9.4 Stage 3 — Mode Selector training (20 epochs)

All upstream components frozen. The Selector is trained to pick the highest-PDMS trajectory from a fixed set of 80 candidates per scene (10 anchors × 8 samples) generated by the Stage 2-B checkpoint.

**Pass criterion**: top-1 selection accuracy beats random by ≥ 30%.

---

## 10. Inference Pipeline

```
2VA input → Emu3 8B forward
         → Joint Attention
         → Action Expert (per anchor):
              ├── Anchored FM head: x_0^k = a^k (no noise) → 2-step ODE → τ^k
              └── Mixture Weight head: ŝ^k
         → Mode Selector picks τ* from {τ^k}
         → Output τ*
```

**Latency budget on H100** (approximate, target，v2.2 调整 N_anchor=20)：

| Component | Time |
|-----------|------|
| Emu3 backbone forward (2VA, 8B, KV cache) | ~125 ms |
| SharedLayer action-side（per 推理 forward） | included |
| Action Expert × 20 anchors × 2 ODE steps（共 40 次小 forward，可在 anchor 维 batch） | ~40 ms |
| Mode Selector（VLM-conditioned，~10M params，K=160 candidates） | ~25 ms |
| Other overhead（tokenization、PDM 后处理） | ~20 ms |
| **Total** | **~210 ms** |

`vlm.lm_head` 不参与推理路径（既不跑 visual-token 自回归也不跑 action 解码），因此推理成本被 VLM transformer forward 主导。Action Expert 的 40 次小 forward 可以把 anchor 维堆到 batch 维一次跑掉，实际开销远小于"40 次串行调用"。

---

## 11. Mathematical Specification (v2.2 总表，与代码同向)

| Quantity | Formula |
|---|---|
| Anchor library | $\{a^k\}_{k=1}^{20}$, K-Means on navtrain GT (x,y only) |
| Positive anchor | $k^+ = \arg\min_k \|a^k_{(:,:2)} - \tau_{GT,(:,:2)}\|_2$ |
| Anchored noise | $\epsilon_{\text{mul}} \sim \mathcal{N}(0, 0.04^2 I_2)$ — 2 scalars (long, lat), broadcast to N_f waypoints, applied only on (x,y) channels |
| **Anchored noise端点** (t=1) | $x_1^{k} = a^k \odot (1 + \epsilon_{\text{mul}})$（heading 通道不变） |
| **Linear path** (t=0 是 GT) | $x_t^k = (1 - t)\, \tau_{GT} + t\, x_1^k$ |
| **Target vector field** | $v^* = x_1^{k^+} - \tau_{GT}$（与代码 `noise - data` 同向） |
| Predicted field | $v_\phi(x_t^k, t, e^k, F_{\text{vlm}})$ |
| Anchor embedding | $e^k = \text{MLP}_a(\text{flatten}(a^k))$，作为额外 prefix token |
| Mixture weight | $\hat s^k = \text{MLP}_s(\text{Pool}(\text{anchor token hidden}))$ |
| FM IL loss | $\mathcal{L}_{FM\text{-}IL} = \mathbb{E}_t\|v_\phi - v^*\|^2$（仅正 anchor $k^+$ 反传） |
| BCE loss | $\mathcal{L}_{BCE} = \sum_k \text{BCE}(\hat s^k, \mathbf{1}[k=k^+])$ |
| **Stochastic ODE step** (multiplicative) | $z_{t-\Delta t}^{\text{mean}} = z_t + \Delta t\, v_\phi$；$z_{t-\Delta t} = z_{t-\Delta t}^{\text{mean}} \odot (1 + \epsilon_{\text{step}})$；$\epsilon_{\text{step}} \sim \mathcal{N}(0, \sigma_{\text{step}}^2 I_2),\ \sigma_{\text{step}} \geq 0.04$ |
| **Log-likelihood**（写在 ε_step 上） | $\log \pi_\theta(\epsilon_{\text{step}}) = -\tfrac{1}{2 \sigma_{\text{logpi}}^2}\|\epsilon_{\text{step}}\|^2 + C,\ \sigma_{\text{logpi}} \geq 0.10$ |
| Reward | $R = -1$ if collision; else $NC \cdot DAC \cdot \tfrac{5\,EP + 5\,TTC + 2\,C}{12}$ |
| Group advantage (intra-anchor) | $A^{k,i} = (r^{k,i} - \mu_k) / (\sigma_k + 10^{-4})$，mean/std over G samples per anchor |
| Truncated advantage | $A^{k,i}_{\text{trunc}} = -1$ if collision/off-road; else $\max(0, A^{k,i}) \cdot \mathbf{1}[r^{k,i} > r_{GT} - 10^{-6}]$ |
| RL loss | $\mathcal{L}_{RL} = -\frac{1}{B \cdot K \cdot T} \sum \exp(\log\pi_\theta - \log\pi_\theta.\text{detach}()) \cdot A^{k,i}_{\text{trunc}} \cdot \gamma^{T-t-1}$ |
| **IL loss in Stage 2-B (vs GT L1)** | $\mathcal{L}_{IL} = \tfrac{1}{D \cdot K \cdot N_f} \sum \|\hat\tau^{k,i}_{(:,:2)} - \tau_{GT,(:,:2)}\|_1$ |
| Adaptive IL weight | $\lambda_{IL} = 0.1$ if any-positive-advantage in batch else $1.0$ |
| Stage 2-B total | $\mathcal{L} = \mathcal{L}_{RL} + \lambda_{IL} \cdot \mathcal{L}_{IL}$ |

---

## 12. Parameter and Compute Profile

### 12.1 Trainable parameter budget (Stage 2-B, v2.2 与代码对齐)

| Module | Trainable params | Notes |
|---|---|---|
| `vlm.*`（Emu3 8B + lm_head） | **0** | 完全 frozen |
| `action_expert.*`（hidden=1024, intermediate=512, 32 层） | **~150–250M（待 Task 0.2 实测）** | 全开 |
| `action_projector.*` / `action_decoder.*` / `tau_emb.*` | ~5M | 全开（与 expert 同步） |
| `state_projector.*` | 0 | frozen（保留 cmd 已学行为） |
| `anchor_embedding.*` | ~3M | 新增 |
| `mixture_weight_head.*` | ~2M | 新增 |
| **Total** | **~150–260M** | v2.0/2.1 的 537M 是错算 |

**R10 兜底**（默认不启用）：Tier-1 LoRA r=8 on `vlm.*.q/v_proj` ≈ 8M，仅在 Phase 3 PDMS 卡死时启用。

**Memory footprint**（per-device，bf16）：
- Frozen `vlm.*` 8B params ≈ 16 GB（forward 用）
- MoVQGAN tokenizer ≈ 1 GB
- Trainable params + Adam optimizer state ≈ 4–8 GB（远小于 v2.0 估算的 30+ GB，因 trainable 缩小到 ~200M）
- Activations / KV cache（2VA seq_len ≈ 1400, B=512 微批次）≈ 15–25 GB
- 总计每卡约 35–50 GB；8× L20 (256 GB) 充裕，可不启用 gradient_accumulation。

### 12.2 GRPO compute cost per step

每个 scene per step：
- 1 × VLM forward（frozen，输出可缓存 KV）
- $N_{\text{anchor}} \times G \times T_{\text{trunc}} = 20 \times 8 \times 10 = 1600$ 次 Action Expert mini-forward（每次只跑 32 层 SharedLayer 的 action 端，attend frozen vlm KV cache，单次极便宜）
- $N_{\text{anchor}} \times G = 160$ 次 PDM scorer 评估
- Pass 2（with_grad）重跑 Action Expert mini-forwards（同样 1600 次，重计算 logπ_new），无需重跑 VLM
- 1 × Action-Expert-only backward（trainable ~200M，比 v2.0 估的 8B+LoRA backward 快得多）

VLM forward 仍是 wall-clock 主项；Pass 1 / Pass 2 共享 VLM forward 是两段式 design 的关键收益。

### 12.3 Total scorer load

navtrain 约 1192 scenes × 160 trajectories per epoch = 190,720 PDM evaluations per epoch（v2.2 的 K=160 是 v2.0 的 2 倍因 N_anchor 改成 20）。16 async workers @ ≥ 200 trajectories/s → ~16 分钟 scorer 工作 per epoch，10 epochs ≈ 2.7 小时纯 scorer 时间，在 GPU 计算的影子里。

---

## 13. Comparison with Baselines

| System | Backbone | Action decoder | Supervision | navtest PDMS |
|---|---|---|---|---|
| TransFuser | ResNet-34 | MLP | IL | 84.0 |
| DiffusionDrive | ResNet-34 | Truncated diffusion | IL | 88.1 |
| DiffusionDrive v2 | ResNet-34 | Truncated diffusion | IL + GRPO | 91.2 |
| DriveVLA-W0 (query-based, paper) | Emu3 8B | Query | IL + WM | 88.4 |
| DriveVLA-W0 (query + anchors, paper) | Emu3 8B | Query × anchors | IL + WM | 90.2 |
| DriveVLA-W0 (AR best-of-6, paper) | Emu3 8B | Autoregressive | IL + WM | 93.0 |
| FM IL baseline B0（**Task 0.3 实测**） | Emu3 8B | Flow Matching | IL（Stage 1 + 2 已完成） | **待测，假设 ≈ 87.2** |
| **This work（Emu3Pi0 + Anchored FM + GRPO，v2.2）** | Emu3 8B（frozen） | **Anchored Flow Matching** | **IL + GRPO**（vlm.* frozen，无显式 WM aux） | **target ≥ B0 + 3.0** |

**与 v2.0 Comparison 的差异**：
- v2.0 把 "WM aux" 列为本工作的 supervision 之一；v2.2 删除——`vlm.lm_head` 在 Stage 2-B 完全 frozen 且不 forward，没有"WM aux loss"参与训练 loss。
- v2.0 给 absolute target ≥ 90.0；v2.2 给相对 target ≥ B0 + 3.0，避免对未独立验证的 87.2 baseline 硬绑定（R11）。

**贡献定位**：DD-v2 的 Anchored GRPO（Intra-anchor advantage + Inter-anchor truncation + multiplicative noise + adaptive IL weight）端到端移植到 DriveVLA-W0 的 Flow Matching action expert 上，让 8B-backbone-frozen 设定下的 FM expert 也能享受 IL → GRPO 的 ~+3 PDMS 提升。Mode Selector 因 backbone 差异从 BEV-conditioned 改为 VLM-hidden-conditioned，是次要但必要的改动。

---

## 14. Limitations and Future Extensions

### 14.1 Known limitations of this design

1. **AR-WM contribution at NAVSIM scale is small.** DriveVLA-W0 Table 3 shows AR-WM provides only +3.6% gain at 70k frames (vs ViT/Diffusion-WM's +19.9%). The World Model is therefore mainly useful as a Stage 1 representation prior; its gradient contribution during Stage 2 is limited.

2. **AR-WM as RL reward is expensive.** Sequential token decoding (256–1024 tokens per future frame) makes model-based RL reward via WM rollouts prohibitive at training time. This Phase 1 plan uses NAVSIM PDM scorer exclusively. Phase 2 model-based RL extension is a research project.

3. **Anchor library is static.** The K-Means anchors are fixed at initialization and never updated. If RL training significantly shifts the trajectory distribution, the anchors may become misaligned. Mitigation: re-cluster anchors after every 5 epochs (not in default plan, optional).

4. **Sample efficiency at scale.** Paper Table 4 shows FM's small-scale advantage reverses at 70M frames. NAVSIM remains within FM's regime, but scaling beyond NAVSIM would require switching to AR Action Expert.

5. **Single front-view camera.** Inherited from DriveVLA-W0's setup. Extending to surround-view multi-camera requires changes upstream of this report's scope.

### 14.2 Phase 2 roadmap

1. **AR Action Expert + token-prefix anchors.** For 70M-frame in-house dataset training where AR Action Expert dominates. Anchored GRPO needs to be redesigned with token-prefix clustering instead of trajectory K-Means.

2. **AR World Model as reward source.** Replace some PDM scorer calls with AR-WM rollouts; check collision in the predicted future frame. Sequential decoding cost is the main hurdle — possible mitigation: use only 1-2 future tokens for collision-relevant features rather than full frame.

3. **Counterfactual reward augmentation.** AR-WM produces interpretable counterfactual sequences (paper §4.6). Use the WM's counterfactual capability to augment training with "what-if" scenarios scored against WM-predicted outcomes. This is a strong fit for the AR-WM architecture.

4. **Compare with retrospective ViT route.** If Qwen2.5-VL + Diffusion-WM checkpoints become available, run a controlled comparison to quantify the small-scale gain that motivated the original ViT recommendation.

---

## Appendix A — Notation

| Symbol | Meaning |
|---|---|
| $L_t, V_t, A_t$ | Text command, image, action at time $t$ (all discrete tokens via Emu3 unified vocab) |
| $S_t$ | Concatenated multi-modal sequence (2VA) |
| $F_L, F_V, F_A$ | Backbone hidden states by modality (continuous) |
| $v_{t+1}^{(i)}$ | $i$-th MoVQGAN visual token of future frame $I_{t+1}$ |
| $a^k$ | $k$-th K-Means anchor trajectory, $\mathbb{R}^{N_f \times 3}$ |
| $e^k$ | Anchor embedding |
| $\tau_{GT}$ | Ground-truth trajectory |
| $\tau^{k,i}$ | $i$-th sampled trajectory from anchor $k$ |
| $x_t^k$ | Trajectory state at flow-matching time $t \in [0, 1]$ |
| $v_\phi$ | Predicted vector field |
| $v^*$ | Target vector field (linear path) |
| $\hat s^k$ | Mixture Weight head output for anchor $k$ |
| $r^{k,i}$ | Reward for $\tau^{k,i}$ |
| $A^{k,i}, A^{k,i}_{\text{trunc}}$ | Group advantage, truncated advantage |
| $\eta$ | ODE stochasticity coefficient (1 train, 0 infer) |
| $\Delta t$ | ODE step size = $1 / T_{\text{trunc}}$ |
| $\sigma_{\text{anchor}}$ | Anchored exploration noise std (default 0.04) |
| $\sigma_{\text{step}}$ | Per-step ODE noise std (clipped to ≥ 0.10) |
| $\gamma$ | RL discount factor (0.8) |
| $N_{\text{anchor}}, G, T_{\text{trunc}}, T_{\text{infer}}$ | 20, 8, 10, 2 |
| $\sigma_{\text{anchor}}, \sigma_{\text{step}}, \sigma_{\text{logpi}}$ | 0.04, 0.04 (min clip), 0.10 (min clip) |
| $\lambda_a$ (Stage 2-A anchor warmup) | 0→1 over 500 steps |
| $\lambda_{IL}$ (Stage 2-B adaptive) | 0.1 if has_positive else 1.0 |

---

## Appendix B — File Mapping (v2.2 与既有仓库对齐)

| Spec section | Code module | Test file |
|---|---|---|
| §3.1 Emu3 backbone | [reference/Emu3/emu3/mllm/modeling_emu3.py](reference/Emu3/emu3/mllm/modeling_emu3.py)（inherited） | `tests/test_backbone_smoke.py` |
| §3.2 R10 兜底 LoRA | `utils/rl_modules/lora_setup.py`（仅在兜底启用时新增） | `tests/test_lora_param_count.py` |
| §4 AR-WM (= vlm.lm_head) | 同 §3.1，不需独立模块 | `tests/test_frozen_vlm_hash.py` |
| §5 SharedLayer | [reference/Emu3/emu3/mllm/modeling_emu3.py:1726-2003](reference/Emu3/emu3/mllm/modeling_emu3.py#L1726-L2003)（inherited） | `tests/test_shared_layer_mask_with_anchor.py` |
| §6.1 Anchor library | `models/policy_head/anchor_kmeans.py`（new） | `tests/test_anchor_silhouette.py` |
| §6.2 Anchor embedding | `models/policy_head/anchor_embedding.py`（new） | `tests/test_zero_anchor_compat.py` |
| §6.3 Anchored path | `models/policy_head/anchored_flow_path.py`（new；扩展 [noise_schedulers.py](models/policy_head/noise_schedulers.py)） | `tests/test_anchored_flow_path.py` |
| §6.4 Vector field head | 复用现有 `action_decoder`（[modeling_emu3.py:1981-1982](reference/Emu3/emu3/mllm/modeling_emu3.py#L1981-L1982)） | `tests/test_fm_head.py` |
| §6.5 ODE samplers | `models/policy_head/stochastic_ode_sampler.py`（new） | `tests/test_stochastic_sampler_logpi.py` |
| §6.6 log π | 同上文件，与 sampler 同接口 | 同上 |
| §7 Mixture Weight | `models/policy_head/mixture_weight_head.py`（new） | `tests/test_mixture_weight.py` |
| §8 Mode Selector（VLM-conditioned） | `models/policy_head/mode_selector.py`（new，重设计） | `tests/test_selector.py` |
| §9.3 GRPO loss | `utils/rl_modules/grpo_loss.py`（new，DD-v2 移植） | `tests/test_grpo_loss.py` |
| §9.3 Rollout | `utils/rl_modules/rollout_collector.py`（new） | `tests/test_rollout_shape.py` |
| §11 Reward wrapper | `utils/rl_modules/pdm_reward_wrapper.py`（封装 [inference/navsim/](inference/navsim/) 内的 PDM scorer） | `tests/test_reward_wrapper.py` |
| 训练入口 | `utils/train_grpo_stage2a.py` / `utils/train_grpo_stage2b.py` / `utils/train_grpo_stage3.py`（new） | `tests/test_e2e_smoke.py` |

---

## Appendix C — Diff from v2.0 (code-level reconciliation, v2.2)

| § | v2.0 | v2.2 |
|---|------|------|
| 2.1 | "AR World Model ~250M, token-head gradient-stopped" | 不存在独立 WM 模块；AR-WM = `vlm.lm_head` 上的 visual-token CE，整个 `vlm.*` frozen |
| 2.1 | "Action Expert 500M" | 实际 ~150–250M（hidden=1024, intermediate=512, 32 层；待 Task 0.2 实测） |
| 2.1 | "Trainable ~537M (LoRA + Expert + heads)" | 实际 ~155–260M（Expert 主体 + 5M 现有小头 + 5M 新 heads，无 LoRA） |
| 3.2 | "LoRA r=16, ~32M params" | 默认 vlm 完全 frozen；R10 兜底 LoRA r=8 ≈ 8M（v2.0 数字偏大 4×） |
| 4 | "Optional β·L_WM-AR auxiliary loss" | 删除——与"vlm 完全 frozen + 不 forward lm_head" 互斥 |
| 5 | "Symmetric Joint Attention: Q=[Q_VLA;Q_AE], K=[K_VLA;K_AE], V=[V_VLA;V_AE]" | 实际是 SharedLayer 非对称耦合：action expert Q 同时 attend (action KV ⊕ vlm KV)；VLM 不变 |
| 6.1 | $N_{anchor}=10$ | $N_{anchor}=20$（与 DD-v2 默认 `ego_fut_mode=20` 对齐） |
| 6.3 | $x_t = (1-t)\, x_0^{anchor} + t\, \tau_{GT}$（t=0 是 anchor） | $x_t = (1-t)\, \tau_{GT} + t\, x_1^{anchor}$（t=0 是 GT，与 [noise_schedulers.py](models/policy_head/noise_schedulers.py) 同向） |
| 6.4 | $v^* = \tau_{GT} - x_0^{anchor}$ | $v^* = x_1^{anchor} - \tau_{GT}$（符号反转，与 `noise - data` 同向） |
| 6.5 | Additive Gaussian per-element step noise: $x + \Delta t v + \sqrt{\eta\Delta t}\xi$ | Multiplicative `(1+ε_step)`，ε_step 仅 (long, lat) 2 标量（DD-v2 §3.1 反对 additive） |
| 6.6 | log π over $x_{t+\Delta t}$ | log π over $\epsilon_{step}$（与 multiplicative 形式一致） |
| 7 | Anchor injection "concatenated at every layer" | 仅在 prefill 注入一个 anchor token（与 state_token 并列），后续 32 层自然传播 |
| 8 | "Structurally unchanged from DD-v2"（BEV+map+agent） | VLM-conditioned selector：trajectory MLP → cross-attn(VLM hidden) → self-attn → score；~10M params |
| 9.2 | Stage 2-A produces frozen reference policy for KL | 取消 reference policy；KL 转 Phase 7.5 ablation |
| 9.3 | $L_{IL}$ 对照 frozen reference | $L_{IL}$ = vs GT trajectory L1（DD-v2 [model_rl.py:1105-1112](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1105-L1112)） |
| 9.3 | Pass criterion 87.2 / 89.0 / 90.0 硬阈值 | 改为相对 B0：B0 ± 0.5（Task 0.3）/ B0 − 1.2（Stage 2-A）/ B0 + 2.0（Stage 2-B）/ B0 + 3.0（Stage 3） |
| 10 | "Action Expert × 10 anchors × 2 ODE steps ~30 ms" | "× 20 anchors × 2 ~40 ms"（v2.2 N_anchor 改为 20） |
| 12.1 | "8B base + LoRA + Expert + heads = 537M trainable" | ~155–260M trainable（无 LoRA） |
| Appendix B | 顶层新建 `drivevla_w0_grpo/` 树 | 沿用既有仓库结构 `models/policy_head/`、`utils/`、`scripts/scripts_train/` |

**核心变化**：v2.2 是 v2.0 的代码级修订版——所有数学方法学（GRPO / intra-anchor advantage / inter-anchor truncation / multiplicative noise / adaptive IL weight）保持不变，但每个 design point 都重新校准到本仓库实际代码可执行的形式。

---

*End of technical report — v2.2*
