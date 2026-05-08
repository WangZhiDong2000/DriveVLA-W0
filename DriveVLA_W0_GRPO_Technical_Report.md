# Technical Report
## DriveVLA-W0 + Flow Matching GRPO: Model Architecture and Component Specification

> **Companion document to the implementation plan**
> Scope: NAVSIM-scale Phase 1 / Flow Matching Action Expert / GRPO via DiffusionDrive v2 RL stack
> **Backbone: Emu3 (8B) + AR World Model** (revised from v1.0 ViT/Diffusion-WM)
> Version: 2.0

---

## Abstract

This report specifies the model architecture and component-level details of a hybrid system that integrates DiffusionDrive v2's Anchored GRPO reinforcement learning into DriveVLA-W0's MoE-based Vision-Language-Action stack. The system uses **DriveVLA-W0 (VQ)** — Emu3 8B backbone with an AR World Model — as the pretrained Stage 1 foundation. Above this we install an *Anchored* Flow Matching Action Expert under GRPO supervision. Two new components are introduced inside the Action Expert: an Anchored Flow Matching head that predicts a vector field conditioned on driving-intent anchors, and a Mixture Weight head that produces per-anchor scores for inference-time selection. The downstream Mode Selector follows DiffusionDrive v2's Coarse-to-Fine design unchanged. **The combination of Emu3 + AR-WM + Flow Matching Action Expert is a paper-validated configuration of DriveVLA-W0 (Table 4 at 70k–700k frames); the novel contribution of this work is the Anchored GRPO supervision applied on top.**

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

### 2.1 Component inventory

| Component | Source | Params | Trainable in Stage 2-B | Role |
|-----------|--------|--------|------------------------|------|
| VLA backbone | DriveVLA-W0 (Emu3 8B) | ~8B | LoRA only (~32M) | Multi-modal context encoder |
| AR World Model | DriveVLA-W0 | ~250M | Token head gradient-stopped | Auxiliary representation supervision |
| Action Expert MoE | DriveVLA-W0 (Flow Matching variant) | 500M | Yes, full | Vector field predictor |
| Anchored FM head | **New** | ~5M | Yes, full | Anchor → conditioning |
| Mixture Weight head | **New** | ~2M | Yes, full | Per-anchor logit |
| Mode Selector | DiffusionDrive v2 | ~50M | No (Stage 3 only) | Top-1 trajectory pick |

Total trainable parameters in Stage 2-B: ≈ 540M (Action Expert 500M + LoRA 32M + new heads 7M).

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

### 3.2 LoRA adaptation

The 8B backbone is **not** fully fine-tuned in Stage 2. LoRA adapters are inserted on attention `q_proj` and `v_proj` layers:

- Rank $r = 16$
- $\alpha = 32$
- LR = $2 \times 10^{-4}$ (same as Action Expert)
- Trainable params via LoRA: ~32M (vs. 8B full)

Reasons for LoRA over full fine-tune:
1. Preserves the World Model representations learned in Stage 1 (which take significant compute to retrain).
2. Reduces optimizer state memory by >30 GB.
3. Adequate capacity to adapt to 2VA input format and modulate features for the new Anchor Embedding pathway.

---

## 4. AR World Model (Auxiliary)

Inherited unchanged from DriveVLA-W0. The AR World Model is a transformer head that predicts the next discrete visual token of the future image $V_{t+1}$ conditioned on current VLA features.

$$\mathcal{L}_{WM\text{-}AR} = -\sum_{i} \log p(v_{t+1}^{(i)} \mid v_{t+1}^{(<i)},\ F_V^t,\ F_A^t)$$

where $v_{t+1}^{(i)}$ is the $i$-th visual token in the MoVQGAN tokenization of $I_{t+1}$.

**Stage 2-B treatment**: the AR-WM token prediction (un-embedding) head is **gradient-stopped** to prevent RL gradients from corrupting its representations (Risk R2 mitigation). Optionally, a small auxiliary loss $\beta = 0.01 \cdot \mathcal{L}_{WM\text{-}AR}$ can be added during Stage 2-B if perplexity on held-out frames degrades by more than 10% — the simpler gradient-stop approach is the default.

**The World Model is never used at inference.** Its sequential decoding cost (256–1024 tokens per future frame, plus MoVQGAN decoding) is not a runtime concern in this work; the WM serves purely as a representation regularizer during training.

---

## 5. MoE and Joint Attention

### 5.1 MoE structure (inherited from DriveVLA-W0)

Two experts share the same transformer block topology but operate at different scales:

- **VLA Expert** (the 8B backbone above) — large hidden dim
- **Action Expert** (500M) — much smaller hidden dim, same number of layers $L$

Each layer of the Action Expert has its own LayerNorm and FFN, but the attention is *jointly* computed across both experts via concatenation.

### 5.2 Joint Attention

For each layer:

$$Q = [Q_{\text{VLA}};\ Q_{\text{AE}}],\quad K = [K_{\text{VLA}};\ K_{\text{AE}}],\quad V = [V_{\text{VLA}};\ V_{\text{AE}}]$$

Attention is computed once on the concatenated tensors, then the output is split back to each expert. This creates a tight, symmetric coupling — the Action Expert can attend over both its own queries and the rich context from the VLA Expert in a single operation.

For the Flow Matching variant, the Action Expert's input prefill includes the previous action $A_{t-1}$ as a temporal prior; the new anchor embedding (Section 6) is also injected here.

**Note on hidden state dimension matching**: Emu3's hidden dim (typically 4096 for 8B variants) differs from Action Expert's. Joint Attention uses standard projection layers ($W_Q, W_K, W_V$) per expert that map to a shared attention dim. This is inherited unchanged from DriveVLA-W0 and requires no modification for this work.

---

## 6. Anchored Flow Matching Head (New)

> **This entire section is backbone-agnostic.** It operates in continuous trajectory space (R^{N_f × 3}) and depends only on the conditioning features supplied by the VLA Expert via Joint Attention.

### 6.1 Anchor library

A library of $N_{\text{anchor}} = 10$ trajectory anchors $\{a^k\}_{k=1}^{10}$ is built once, offline, by K-Means clustering all NAVSIM `navtrain` ground-truth trajectories. Each anchor $a^k \in \mathbb{R}^{N_f \times 3}$ represents a distinct driving intent (go straight, turn left, turn right, lane change left, etc.) where $N_f = 8$ waypoints per NAVSIM convention.

This step reuses DiffusionDrive v2's K-Means implementation directly.

### 6.2 Anchor embedding

Each anchor is embedded into the Action Expert's hidden space:

$$e^k = \text{MLP}_a(\text{flatten}(a^k)) \in \mathbb{R}^{d_{\text{AE}}}$$

The anchor embedding is concatenated with the noised trajectory at the input to each Action Expert layer, providing per-anchor conditioning throughout the network.

### 6.3 Anchored straight-line path (training)

For each training scene, the positive anchor is selected as:

$$k^+ = \arg\min_k \|a^k - \tau_{GT}\|_2$$

The anchored starting point is constructed with scale-adaptive multiplicative noise:

$$x_0^{k^+} = a^{k^+} \odot (1 + \epsilon_{\text{mul}}),\quad \epsilon_{\text{mul}} = (\epsilon_{\text{long}}, \epsilon_{\text{lat}}) \sim \mathcal{N}(0, \sigma_{\text{anchor}}^2 I_2)$$

with $\sigma_{\text{anchor}} = 0.04$ (DiffusionDrive v2 default). The noise has only **two scalars** broadcast across all waypoints — longitudinal and lateral — preserving trajectory smoothness.

The flow matching path is the standard linear interpolation:

$$x_t^{k^+} = (1-t) \cdot x_0^{k^+} + t \cdot \tau_{GT},\quad t \in [0, 1]$$

### 6.4 Vector field predictor

The Action Expert's main output is a vector field $v_\phi \in \mathbb{R}^{N_f \times 3}$ predicted at intermediate time $t$:

$$v_\phi = v_\phi(x_t^k,\ t,\ e^k,\ F_{\text{AE}})$$

For a linear path, the **target vector field is constant**:

$$v^*(x_t^{k^+}, t) = \tau_{GT} - x_0^{k^+}$$

The IL training objective for the Anchored FM head is:

$$\mathcal{L}_{FM\text{-}IL} = \mathbb{E}_t\left[\ \|v_\phi(x_t^{k^+},\ t,\ e^{k^+},\ F_{\text{AE}}) - (\tau_{GT} - x_0^{k^+})\|^2\ \right]$$

**Crucial detail**: only the positive anchor $k^+$ contributes to $\mathcal{L}_{FM\text{-}IL}$. Negative anchors do not get reconstruction supervision — this prevents collapse-to-mean and mirrors DiffusionDrive's $y^k = 1$ masking in Eq. (4) of the original paper.

### 6.5 ODE samplers

Two ODE samplers are implemented:

**Training (η = 1, stochastic, $T_{\text{trunc}} = 10$ steps)**:
$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\phi + \sqrt{\eta \Delta t} \cdot \xi_t,\quad \xi_t \sim \mathcal{N}(0, I)$$

The injected per-step noise enables policy-gradient REINFORCE updates and broadens exploration.

**Inference (η = 0, deterministic, 2 steps)**:
$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\phi$$

Two steps suffice because of the anchored start.

### 6.6 Policy log-likelihood

For GRPO, the log-likelihood of the stochastic step is:

$$\log \pi_\theta(x_{t + \Delta t} \mid x_t) = -\frac{1}{2 \sigma_{\text{step}}^2}\|x_{t + \Delta t} - x_t - \Delta t \cdot v_\phi\|^2 + C$$

where $\sigma_{\text{step}} = \max(\sqrt{\Delta t}, 0.10)$ — the lower clip prevents gradient explosion when $\Delta t$ is small (Risk R6 mitigation).

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

## 8. Mode Selector (Inherited from DiffusionDrive v2)

The Mode Selector is **structurally unchanged** from DiffusionDrive v2. Trajectory coordinates serve as queries and pass through:

1. Deformable spatial cross-attention against BEV features (from a small perception encoder)
2. Cross-attention against agent and map queries
3. MLP to produce a scalar score

A two-stage Coarse-to-Fine arrangement is used: the coarse selector picks top-$k$ candidates, the fine selector picks the final winner. Loss is BCE + Margin-Rank Loss (DiffusionDrive v2 Eq. 11).

Training is independent (Stage 3, 20 epochs, all upstream components frozen). Data augmentation:

- Multiplicative Gaussian noise on RL-generated trajectories, std ∈ [0.1, 0.2]
- 1% supplementary trajectories sampled from the GTRS fixed vocabulary

---

## 9. Training Pipeline

### 9.1 Stage 1 — pretrained, inherited

Inherited from DriveVLA-W0. The 8B Emu3 backbone + AR World Model is jointly pretrained on 6VA sequences with:

$$\mathcal{L}_{\text{Stage 1}} = \mathcal{L}_{\text{Action}} + \beta \cdot \mathcal{L}_{WM\text{-}AR}$$

This work *consumes* the resulting checkpoint; **no Stage 1 training is performed**. The checkpoint is downloaded from the DriveVLA-W0 public release.

### 9.2 Stage 2-A — IL warm-up (4k steps)

**Inputs**: Stage 1 checkpoint, anchor library $\{a^k\}$
**Trainable**: LoRA adapters on backbone + Action Expert (full) + Anchored FM head + Mixture Weight head
**Frozen**: Emu3 base weights (LoRA bypassed) + AR-WM token prediction head
**Objective**:

$$\mathcal{L}_{\text{Stage 2-A}} = \mathcal{L}_{FM\text{-}IL} + \mathcal{L}_{BCE}$$

This stage serves three purposes:
1. Adapt the backbone from 6VA to 2VA input format.
2. Initialize the Action Expert and new heads from scratch.
3. Produce a checkpoint that serves as the **frozen reference policy** for KL-style regularization in Stage 2-B.

**Pass criterion**: navtest PDMS ≥ 86.0.

### 9.3 Stage 2-B — GRPO main training (10 epochs)

**Inputs**: Stage 2-A checkpoint as both initialization and frozen reference
**Trainable**: same as Stage 2-A
**Per-step procedure** for each scene in the batch:

1. Backbone forward + Joint Attention (using Stage 1 ckpt + Stage 2-A LoRA)
2. For each anchor $k$, sample $G = 8$ exploration noises $\{\epsilon_{\text{mul}}^i\}$
3. Run stochastic ODE for $T_{\text{trunc}} = 10$ steps → clean trajectories $\{\tau^{k,i}\}$
4. Score each $\tau^{k,i}$ via NAVSIM PDM scorer (parallel pool of 16 workers) → reward $r^{k,i}$
5. Compute Intra-Anchor advantage $A^{k,i} = (r^{k,i} - \text{mean}_k) / \text{std}_k$
6. Apply Inter-Anchor truncation: $A^{k,i}_{\text{trunc}} = -1$ if collision, else $\max(0, A^{k,i})$
7. Compute RL loss:

$$\mathcal{L}_{RL} = -\frac{1}{N_{\text{anchor}} \cdot G \cdot T} \sum_{k}\sum_{i}\sum_{t} \gamma^{t-1}\ \log \pi_\theta(x_{t-1}^{k,i} \mid x_t^{k,i})\ A^{k,i}_{\text{trunc}}$$

8. Compute IL regularization $\mathcal{L}_{IL}$ against the frozen Stage 2-A reference checkpoint
9. Total: $\mathcal{L} = \mathcal{L}_{RL} + 0.1 \cdot \mathcal{L}_{IL}$

**Hyperparameters**: $\gamma = 0.8$, batch size 512, AdamW with cosine schedule, bf16, gradient clip norm 1.0.
**Pass criterion**: navtest PDMS ≥ 89.0 (≥ 2 PDMS over Stage 2-A).

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

**Latency budget on H100** (approximate, target):

| Component | Time |
|-----------|------|
| Emu3 backbone forward (2VA, 8B) | ~125 ms |
| Joint Attention | included |
| Action Expert × 10 anchors × 2 ODE steps | ~30 ms |
| Mode Selector | ~30 ms |
| Other overhead | ~20 ms |
| **Total** | **~205 ms** |

The AR World Model is not used at inference; only the backbone hidden states feed into the Action Expert. This keeps the inference cost dominated by the backbone forward and avoids the slow sequential AR-WM decoding entirely.

---

## 11. Mathematical Specification (Summary Table)

| Quantity | Formula |
|---|---|
| Anchor library | $\{a^k\}_{k=1}^{10}$, K-Means on navtrain GT |
| Positive anchor | $k^+ = \arg\min_k \|a^k - \tau_{GT}\|_2$ |
| Anchored noise | $\epsilon_{\text{mul}} \sim \mathcal{N}(0, 0.04^2 I_2)$ scalar long+lat |
| Anchored start | $x_0^{k} = a^k \odot (1 + \epsilon_{\text{mul}})$ |
| Linear path | $x_t^k = (1-t)\, x_0^k + t\, \tau_{GT}$ |
| Target vector field | $v^* = \tau_{GT} - x_0^{k^+}$ |
| Predicted field | $v_\phi(x_t^k, t, e^k, F_{\text{AE}})$ |
| Anchor embedding | $e^k = \text{MLP}_a(\text{flatten}(a^k))$ |
| Mixture weight | $\hat s^k = \text{MLP}_s(\text{Pool}(F_{\text{AE}}^k))$ |
| FM IL loss | $\mathcal{L}_{FM\text{-}IL} = \mathbb{E}_t\|v_\phi - v^*\|^2$ (positive anchor only) |
| BCE loss | $\mathcal{L}_{BCE} = \sum_k \text{BCE}(\hat s^k, \mathbf{1}[k=k^+])$ |
| WM aux loss (Stage 1, gradient-stopped at Stage 2-B) | $\mathcal{L}_{WM\text{-}AR} = -\sum_i \log p(v_{t+1}^{(i)} \mid \cdots)$ |
| Stochastic step | $x_{t + \Delta t} = x_t + \Delta t\, v_\phi + \sqrt{\Delta t}\, \xi$ |
| Log-likelihood | $\log \pi_\theta = -\frac{1}{2 \sigma^2}\|x_{t+\Delta t} - x_t - \Delta t v_\phi\|^2 + C$ |
| Min std clip | $\sigma_{\text{step}} \geq 0.10$ |
| Reward | $R = -1$ if collision; else $NC \cdot DAC \cdot \tfrac{5\,EP + 5\,TTC + 2\,C}{12}$ |
| Group advantage | $A^{k,i} = (r^{k,i} - \text{mean}_k) / \text{std}_k$ |
| Truncated advantage | $A^{k,i}_{\text{trunc}} = -1$ if collision; else $\max(0, A^{k,i})$ |
| RL loss | $\mathcal{L}_{RL} = -\frac{1}{N \cdot G \cdot T} \sum \gamma^{t-1} \log \pi_\theta \cdot A^{k,i}_{\text{trunc}}$ |
| Stage 2-B total | $\mathcal{L} = \mathcal{L}_{RL} + 0.1 \cdot \mathcal{L}_{IL}$ |

---

## 12. Parameter and Compute Profile

### 12.1 Trainable parameter budget (Stage 2-B)

| Module | Trainable params | Notes |
|---|---|---|
| Emu3 backbone (8B) | ~32M (LoRA only) | Base 8B frozen |
| Action Expert | 500M | Full fine-tune |
| Anchored FM head (Anchor MLP) | ~3M | New |
| Mixture Weight head | ~2M | New |
| **Total** | **~537M** | |

The base Emu3 frozen weights are still loaded in GPU memory for forward pass; with bf16 they consume ~16 GB. The MoVQGAN tokenizer adds ~1 GB. Total GPU memory per device during Stage 2-B training is approximately 45–55 GB (8× L20 = 384 GB total). Batch size 512 is feasible with gradient accumulation if needed.

### 12.2 GRPO compute cost per step

For each scene in a batch:
- 1 × backbone forward (shared across all anchors via cache)
- $N_{\text{anchor}} \times G \times T_{\text{trunc}} = 10 \times 8 \times 10 = 800$ Action Expert mini-forwards (each is small)
- $N_{\text{anchor}} \times G = 80$ PDM scorer evaluations
- 1 × backbone backward (LoRA-only, much cheaper than full)

The backbone forward dominates wall-clock time. The 800 Action Expert mini-forwards are cheap (500M params each) and amortized via batching across the trajectory dimension. **The AR-WM is not invoked at all during Stage 2-B** (gradient-stopped, no forward pass through token head needed unless using auxiliary loss).

### 12.3 Total scorer load

navtrain has 1192 scenes × 80 trajectories per epoch = 95,360 PDM evaluations per epoch. With 16 async workers @ ≥ 200 trajectories/s aggregate throughput, this is ~8 minutes of pure scorer work per epoch — well within budget.

---

## 13. Comparison with Baselines

| System | Backbone | Action decoder | Supervision | navtest PDMS (target) |
|---|---|---|---|---|
| TransFuser | ResNet-34 | MLP | IL | 84.0 |
| DiffusionDrive | ResNet-34 | Truncated diffusion | IL | 88.1 |
| DiffusionDrive v2 | ResNet-34 | Truncated diffusion | IL + GRPO | 91.2 |
| DriveVLA-W0 (query-based) | Emu3 8B | Query | IL + WM aux | 88.4 |
| DriveVLA-W0 (query + anchors) | Emu3 8B | Query × anchors | IL + WM aux | 90.2 |
| DriveVLA-W0 (AR best-of-6) | Emu3 8B | Autoregressive | IL + WM aux | 93.0 |
| **This work (Emu3 + FM + Anchored GRPO)** | Emu3 8B | **Anchored Flow Matching** | **IL + GRPO + WM aux** | **target ≥ 90.0** |

The contribution is the *combination*:
- DriveVLA-W0's AR World Model provides representation supervision that prevents action-only overfitting (paper-validated at multiple scales)
- DiffusionDrive v2's Anchored GRPO raises both the lower and upper bounds of the trajectory distribution
- Flow Matching expert at NAVSIM-scale data is consistent with paper's Table 4 small-scale findings (FM beats Query and AR at 70k frames)

The Anchored GRPO addition is the novel piece. The headline question is whether it can take a Flow Matching expert from ~87 IL baseline to ≥90 PDMS, mirroring DiffusionDrive v2's IL→GRPO improvement on a smaller backbone.

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
| $N_{\text{anchor}}, G, T_{\text{trunc}}$ | 10, 8, 10 |

---

## Appendix B — File Mapping

| Spec section | Code module | Test file |
|---|---|---|
| §3.1 Emu3 backbone | `drivevla_w0/backbone/emu3.py` (inherited) | `tests/test_backbone_smoke.py` |
| §3.2 LoRA | `training/lora_setup.py` | `tests/test_lora_param_count.py` |
| §4 AR World Model | `drivevla_w0/world_model/ar_wm.py` (inherited) | `tests/test_wm_grad_stop.py` |
| §5 Joint Attention | `drivevla_w0/moe/joint_attention.py` (inherited) | `tests/test_joint_attn_shape.py` |
| §6.1 Anchor library | `rl_modules/anchor_kmeans.py` (DD-v2) | `tests/test_anchor_silhouette.py` |
| §6.2 Anchor embedding | `action_expert/anchor_embedding.py` (new) | `tests/test_anchor_emb.py` |
| §6.3 Anchored path | `action_expert/anchored_flow_path.py` (new) | `tests/test_anchored_fm.py` |
| §6.4 Vector field head | `action_expert/anchored_flow_matching.py` (new) | `tests/test_fm_head.py` |
| §6.5 ODE samplers | `action_expert/ode_sampler.py` (new) | `tests/test_ode_det_stoch.py` |
| §6.6 log π | `action_expert/log_pi.py` (new) | `tests/test_log_pi.py` |
| §7 Mixture Weight | `action_expert/mixture_weight_head.py` (new) | `tests/test_mixture_weight.py` |
| §8 Mode Selector | `rl_modules/mode_selector.py` (DD-v2) | `tests/test_selector.py` |
| §9.3 GRPO loss | `rl_modules/grpo_loss.py` (DD-v2 + glue) | `tests/test_grpo_loss.py` |
| §9.3 Rollout | `rl_modules/rollout_collector.py` (DD-v2) | `tests/test_rollout_shape.py` |
| §11 Reward wrapper | `rl_modules/pdm_reward_wrapper.py` (DD-v2) | `tests/test_reward_wrapper.py` |

---

## Appendix C — Diff from v1.0 (ViT/Diffusion-WM)

| § | v1.0 | v2.0 (this version) |
|---|------|---------------------|
| 1 | Qwen2.5-VL 7B + Diffusion-WM | Emu3 8B + AR-WM |
| 2.1 | LoRA ~30M, total ~535M trainable | LoRA ~32M, total ~537M trainable |
| 3.1 | ViT continuous patch features | MoVQGAN discrete visual tokens; backbone hidden states still continuous |
| 4 | Latent diffusion (MSE on noise) | Next-token prediction (cross-entropy) |
| 4 mitigation | Decoder gradient-stop | Token-head gradient-stop |
| 9.1 | Reuse pretrained checkpoint | Same — but **public Emu3 ckpt available, ViT not** |
| 10 | ~190 ms latency | ~205 ms latency (8B vs 7B) |
| 12.1 | 7B base in memory | 8B base + MoVQGAN tokenizer in memory |
| 13 | "Comparable to FM IL ~87.2 baseline" | Comparable to query-based 88.4 / no published FM-NAVSIM baseline |
| 14 | WM-as-reward easy (Diffusion is differentiable) | WM-as-reward expensive (AR sequential) |

**The Anchored GRPO methodology, mathematical specification, and TDD task structure are identical between v1.0 and v2.0.** The change is exclusively in the Stage 1 foundation.

---

*End of technical report — v2.0*
