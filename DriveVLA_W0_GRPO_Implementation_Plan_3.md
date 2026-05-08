# DriveVLA-W0 + DiffusionDrive v2 GRPO 详细实施计划

> **Phase 1 / NAVSIM 规模 / Flow Matching Action Expert**
> **Backbone: Emu3 8B + AR World Model（FROZEN）**
> **Starting point: `liyingyan/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2` HuggingFace ckpt**
> Version: 2.1

---

## 0. v2.1 修订说明

### 0.1 关键变更

v2.1 基于公开发布的 `Emu3_Flow_Matching_Action_Expert_PDMS_87.2` checkpoint（HuggingFace `liyingyan/DriveVLA-W0`）作为起点。这是一个已经完成 Stage 2 IL 训练的 ckpt，PDMS 87.2 即为 paper-validated FM expert IL 基线数值。

**关键工程简化**：
1. ✅ 跳过 Phase 0（无需复现 baseline）—— ckpt 已发布即为 baseline
2. ✅ Backbone (Emu3 8B) 与 World Model 全程 **完全 frozen**，零参数修改
3. ✅ 仅对 Flow Matching Action Expert 做 anchor 结构改造
4. ✅ 不再使用 LoRA，无需调研 Emu3 attention 层命名
5. ✅ 6VA→2VA 适配在 ckpt 中已完成

**实质修改范围**：仅 500M Action Expert + 新增 ~7M anchor/mixture heads。

### 0.2 v2.0 → v2.1 主要变更

| 项 | v2.0 | v2.1 |
|---|---|---|
| Phase 0 | 1-2 days 环境+基线 | **跳过** |
| Stage 1 ckpt 来源 | DriveVLA-W0 公开 release | **`Emu3_Flow_Matching_Action_Expert_PDMS_87.2` HF release** |
| ckpt 类型 | Stage 1（仅 backbone+WM） | **Stage 2 完整 ckpt（含已训 FM expert）** |
| Backbone fine-tune | LoRA(r=16) | **完全 frozen** |
| WM 处理 | Token head frozen + 不 forward | **整个 WM frozen + 不 forward**（更严格） |
| IL warmup 性质 | 从零训练 Action Expert | **adapt 已训 FM expert 到 anchored 结构** |
| Risk 数 | 8 项 | **6 项**（解决 R5/R7/R8，新增 R9/R10） |
| Trainable params | ~537M (LoRA + Expert) | **~507M（仅 Expert + 新 heads）** |
| 总时长 | 5-6 周 | **4-5 周**（省 Phase 0 + 简化 R5/R7/R8） |

### 0.3 IL baseline 锁定

Phase 3 的 IL warmup 目标值现在**有了精确锚点**：原始 ckpt 的 PDMS 是 87.2。Anchor 结构适配后允许小幅下降（~1 PDMS），但应保持在 ≥ 86.0 PDMS。

---

## 1. 关键设计（与 v2.0 一致）

### 1.1 Anchored Flow Matching 严格表述

设训练样本 $(z, \tau_{GT})$，K-Means 得 $N_{anchor}=10$ 个聚类中心 $\{a^k\}_{k=1}^{N_{anchor}}$。

**正样本 anchor**：$k^+ = \arg\min_k \|a^k - \tau_{GT}\|_2$

**Anchored 直线路径**：
- 起始：$x_0^{k^+} = a^{k^+} \odot (1 + \epsilon_{mul}), \quad \epsilon_{mul} = (\epsilon_{long}, \epsilon_{lat}) \sim \mathcal{N}(0, \sigma_{anchor}^2 I)$
- 路径：$x_t^{k^+} = (1-t) \cdot x_0^{k^+} + t \cdot \tau_{GT}$
- **目标向量场（直线路径下为常向量）**：$v^*(x_t^{k^+}, t) = \tau_{GT} - x_0^{k^+}$
- **IL Loss**：$L_{FM\text{-}IL} = \mathbb{E}_t \left[ \| v_\phi(x_t^{k^+}, t, z, a^{k^+}) - (\tau_{GT} - x_0^{k^+}) \|^2 \right]$

**Mixture Weight Head**：
- 结构：MLP，输入 anchor-conditioned feature → 标量 logit $\hat s^k$
- $L_{BCE} = \sum_k \text{BCE}(\hat s^k, \mathbf{1}[k=k^+])$

**Stage 2-A 总 IL loss**：$L_{IL} = L_{FM\text{-}IL} + L_{BCE}$

### 1.2 GRPO 阶段（与 v2.0 一致）

略，参见 v2.0 §1.2。

### 1.3 Reward 严格定义（与 v2.0 一致）

$$R(\tau) = \begin{cases} -1 & \text{if collision} \\ NC \cdot DAC \cdot \frac{5\,EP + 5\,TTC + 2\,C}{12} & \text{otherwise} \end{cases}$$

---

## 2. 顺序子任务（TDD-style）

### Task 0 — Checkpoint 解析与组件分离（0.5 day）

> 这是计划的实际起点。Phase 0 的环境搭建 + 基线复现已跳过。

#### Task 0.1 — 下载与加载
- **Action**：从 HuggingFace 下载 `liyingyan/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2`。检查 ckpt 文件结构（state dict keys、config files）。
- **Verification**：`torch.load` 成功；记录所有 top-level key prefix。
- **Pass criteria**：ckpt 完整，可读取。

#### Task 0.2 — 组件分离
- **Action**：将 ckpt 的 state_dict 按 prefix 分离为三部分：
  ```python
  full_ckpt = torch.load("checkpoint.pt", map_location="cpu")
  
  backbone_state = {k.replace("backbone.", "", 1): v 
                    for k, v in full_ckpt.items() if k.startswith("backbone.")}
  wm_state       = {k.replace("world_model.", "", 1): v 
                    for k, v in full_ckpt.items() if k.startswith("world_model.")}
  expert_state   = {k.replace("action_expert.", "", 1): v 
                    for k, v in full_ckpt.items() if k.startswith("action_expert.")}
  ```
  实际 prefix 命名以 release 为准。**保存为三个独立的 `.pt` 文件**便于后续单独加载。
- **Verification**：三部分参数量加总等于完整 ckpt；每部分 load 到对应 nn.Module 时 `missing_keys=[]`、`unexpected_keys=[]`。
- **Pass criteria**：三个独立 ckpt 都可干净加载，无 mismatch。

#### Task 0.3 — Frozen 加载验证
- **Action**：加载 backbone 与 WM 后立即 `requires_grad_(False)` 并 `model.eval()`。同时验证 PDMS 87.2 复现：用 `inference.py` 跑 navtest，确认与 release 文档一致。
- **Verification**：navtest PDMS 在 87.2 ± 0.3 范围内；forward 一次后 backbone/WM 参数 hash 不变。
- **Pass criteria**：PDMS 复现 + 参数 hash 校验通过。

---

### Phase 1：DiffusionDrive v2 RL 模块抽取（3-5 days）

> **此 Phase 完全不受任何变更影响**，DD-v2 RL 模块在轨迹空间工作，与 backbone/ckpt 形态解耦。

#### Task 1.1 — `anchor_kmeans.py`
- **Action**：从 DD-v2 复制 K-Means anchor 聚类逻辑，适配 NAVSIM 8 waypoints 格式。接口：`fit(trajectories) → np.ndarray[N_anchor, N_f, 3]`
- **Verification**：对 navtrain 1192 GT 轨迹聚类，可视化 cluster 中心 + 50 条样本。
- **Pass criteria**：silhouette score > 0.4；可视化能区分直行/左转/右转/换道意图。

#### Task 1.2 — `multiplicative_noise.py`
- **Action**：复用 DD-v2 的 $\epsilon_{mul} = (\epsilon_{long}, \epsilon_{lat})$，接口 `apply_noise(traj, std) → traj_noisy`。
- **Verification**：unit test + 加性 vs 乘性 noise 对比可视化。
- **Pass criteria**：unit test 通过；输出无 jagged path。

#### Task 1.3 — `intra_anchor_advantage.py`
- **Action**：复用 DD-v2 的 group advantage：$A^{k,i} = (r^{k,i} - \text{mean}_k) / \text{std}_k$。
- **Verification**：合成 G=8 reward → advantage mean=0, std=1；std=0 边界情况加 epsilon。
- **Pass criteria**：unit test 通过。

#### Task 1.4 — `inter_anchor_truncated.py`
- **Action**：复用 DD-v2 truncation：碰撞→-1，其他负值→0。
- **Verification**：合成 mixed reward + collision flag。
- **Pass criteria**：unit test 通过。

#### Task 1.5 — `pdm_reward_wrapper.py`
- **Action**：包装 NAVSIM PDM scorer 为 batched reward function。
- **Verification**：feed GT → R≈0.95+；feed collision → R=-1；feed off-road → R 显著下降。
- **Pass criteria**：3 种 case 都符合预期。

#### Task 1.6 — `rollout_collector.py`
- **Action**：复用 DD-v2 rollout 收集器，返回 `RolloutBatch(trajectories, log_probs, advantages, anchors)`。
- **Verification**：feed 4 scenes，N_anchor=10, G=8 → 320 个 tuple。
- **Pass criteria**：shape 正确，log_prob 数值有限。

#### Task 1.7 — `mode_selector.py`
- **Action**：复用 DD-v2 Coarse-to-Fine selector + BCE + Margin-Rank loss。
- **Verification**：合成 (trajectory, score) 数据集训练 100 steps。
- **Pass criteria**：loss 单调下降；top-1 accuracy 提升。

---

### Phase 2：FM Action Expert Anchored 改造（5-7 days）

> **核心实施任务**——在已训练好的 FM expert 上添加 anchor 结构，**不动 backbone 和 WM**。

#### Task 2.1 — Action Expert 模块识别与隔离
- **Action**：定位 ckpt 中 Action Expert 的 transformer block 结构。识别 FM head 的输入维度（应能接受 noised trajectory $x_t$ + time embedding $t$ + scene context $z$）和输出维度（vector field $v_\phi$）。
- **Verification**：导出 Expert 模块的 nn.Module 结构图；尝试 forward pass 喂入 dummy `(x_t, t, z)` 确认输出 shape 匹配 `(N_f, 3)`。
- **Pass criteria**：能独立 forward Action Expert 而不依赖完整 ckpt 结构。

#### Task 2.2 — Anchor Embedding 注入
- **Action**：新增 `anchor_embedding.py`：
  ```python
  class AnchorEmbedding(nn.Module):
      def __init__(self, n_waypoints, d_model):
          super().__init__()
          self.mlp = nn.Sequential(
              nn.Linear(n_waypoints * 3, d_model),
              nn.SiLU(),
              nn.Linear(d_model, d_model),
          )
      def forward(self, anchor):  # anchor: (B, N_f, 3)
          return self.mlp(anchor.flatten(1))  # → (B, d_model)
  ```
  注入位置：与 time embedding 同级，加性融合到 Action Expert 第一层输入。
- **Verification**：不同 anchor → 不同输出 feature；当 anchor=0 时输出应近似原始 FM expert 行为。
- **Pass criteria**：smoke test + zero-anchor 行为兼容性。

#### Task 2.3 — Anchored Flow Path
- **Action**：实现 `anchored_flow_path.py`，公式 §1.1。
- **Verification**：t=0 时 $x \approx a^k$（norm diff < 0.1）；t=1 时 $x = \tau_{GT}$（exact）。
- **Pass criteria**：unit test 通过。

#### Task 2.4 — Mixture Weight Head
- **Action**：新增 `MixtureWeightHead`：MLP 输入 anchor-conditioned hidden state → 标量 logit。结构与 Anchor Embedding 平行。
- **Verification**：output shape；softmax sum=1；gradient flow 正常。
- **Pass criteria**：unit test 通过。

#### Task 2.5 — ODE Sampler
- **Action**：实现 `train_sampler(η=1, T=10)` + `infer_sampler(η=0, T=2)`。
- **Verification**：η=0 deterministic check；η=1 stochastic check；mean 一致性。
- **Pass criteria**：3 项性质验证通过。

#### Task 2.6 — Log Policy
- **Action**：实现 $\log \pi_\theta$，min std clip ≥ 0.10。
- **Verification**：100 batches 无 NaN；log π 数值有限。
- **Pass criteria**：unit test + smoke 训练通过。

---

### Phase 3：IL 适配训练 Stage 2-A（2-3 days）

> v2.0 是"从零训练 Expert"，v2.1 是"已有 PDMS 87.2 expert 适配 anchor 结构"。**步数大幅缩短**（4k → 2k），因为只需让新 heads 收敛。

#### Task 3.1 — 训练脚本框架
- **Action**：写 `training/stage2a_anchor_adaptation.py`。**Trainable**：Action Expert（全参）+ Anchor Embedding + Mixture Weight Head（约 507M 参数）。**Frozen**：Emu3 backbone + 整个 World Model + tokenizers。
- **Verification**：单 step forward+backward 通过；trainable param count ≈ 507M；frozen param 的 grad=None。
- **Pass criteria**：smoke test 通过。

#### Task 3.2 — 渐进式 Anchor 引入（v2.1 关键风险缓解）
- **Action**：分两阶段缩短发散风险：
  - 阶段 a（前 500 steps）：Anchor Embedding 输出乘以 warmup 系数 $\lambda_a$，从 0 线性增到 1。这让 expert 起步时几乎沿用原始 PDMS 87.2 行为，逐渐引入 anchor conditioning。
  - 阶段 b（500 → 2000 steps）：$\lambda_a = 1$ 全开，正常训练。
- **Verification**：每 200 steps eval navtest PDMS。**初期 navtest PDMS 应保持在 86 以上**（因 anchor 影响小），后期向上爬。
- **Pass criteria**：训练曲线无大幅下跌；最终 PDMS ≥ 86.0。

#### Task 3.3 — IL Loss 实现
- **Action**：组装 $L_{IL} = L_{FM\text{-}IL} + L_{BCE}$。
- **Verification**：初始 loss 量级合理；100 steps 单调下降。
- **Pass criteria**：loss 收敛趋势正常。

#### Task 3.4 — 完整 2k steps 适配训练
- **Action**：跑完整训练。
- **Verification**：每 500 steps eval navtest，绘制曲线。
- **Pass criteria**：`final navtest PDMS ≥ 86.0`（允许相对原 ckpt 最多 1.2 PDMS 退化，换取 anchor 多模态结构）。

#### Task 3.5 — Reference Model 保存
- **Action**：保存 Stage 2-A final ckpt 到 `reference/stage2a_ref.pt`，frozen，仅 forward。
- **Verification**：reference model 加载 + forward 无误，所有参数 `requires_grad=False`。
- **Pass criteria**：可在 GRPO 阶段被调用作为 KL 正则锚点。

---

### Phase 4：GRPO 训练 Stage 2-B（7-10 days）

> 与 v2.0 流程完全一致，唯一差别是 trainable 参数集合（无 LoRA）。

#### Task 4.1 — Rollout 集成
- **Action**：把 Phase 1 的 `rollout_collector.py` 接入训练循环。
- **Verification**：单 batch 4 scenes，rollout 时间 < 5s。
- **Pass criteria**：throughput 满足要求。

#### Task 4.2 — 异步并行 Scorer
- **Action**：multiprocessing pool（16 workers）异步评估 PDM。
- **Verification**：throughput ≥ 200 trajectory/s；scorer 占比 < 30%。
- **Pass criteria**：满足两项指标。

#### Task 4.3 — GRPO Loss 集成
- **Action**：组装 $L = L_{RL} + 0.1 \cdot L_{IL}$，$L_{IL}$ 对比 Stage 2-A reference。
- **Verification**：loss 项数值平衡；gradient norm < 1.0。
- **Pass criteria**：数值合理 + gradient 不爆炸。

#### Task 4.4 — 1 Epoch Smoke Test
- **Action**：1 epoch GRPO 训练。
- **Verification**：navtest PDMS ≥ 86.0（与 Phase 3 一致或更高）。
- **Pass criteria**：稳定性通过。

#### Task 4.5 — 完整 10 Epochs GRPO
- **Action**：完整训练 + 每 epoch navtest + early stop（patience=2）。
- **Verification**：曲线监控 + best ckpt 保存。
- **Pass criteria**：`final navtest PDMS ≥ 89.0`（IL baseline + 3，复制 DD-v2 改善幅度）。

---

### Phase 5：风险缓解（与 Phase 4 并行执行）

#### Task 5.1 — Risk R1: Anchor σ 调参
- **Action**：grid search $\sigma_{anchor} \in \{0.02, 0.04, 0.08, 0.16\}$，每个 σ 跑 2 epochs。
- **Pass criteria**：选最佳 σ 进入完整训练。

#### Task 5.2 — Risk R2: WM/Backbone 表示保护（v2.1 简化）
- **Action**：v2.1 下 backbone 和 WM 都 100% frozen + 不参与 forward backward path。**唯一保护工作**：用 `requires_grad_(False)` 锁住 + 在训练循环内 assert 参数 hash 不变。
- **Verification**：训练前后 backbone/WM 参数 hash 完全一致。
- **Pass criteria**：hash 校验通过 + R2 自动消解（无可能退化路径）。

#### Task 5.3 — Risk R3: Scorer 瓶颈
- **Action**：详见 Task 4.2。16-worker async pool + cache。
- **Pass criteria**：scorer 占比 < 30%。

#### Task 5.4 — Risk R4: navtrain 过拟合
- **Action**：每 epoch navtest 验证 + early stop + λ 可增到 0.2。
- **Pass criteria**：navtrain-navtest gap < 3 PDMS。

#### Task 5.5 — Risk R6: log-likelihood 数值不稳
- **Action**：min log-var std ≥ 0.10 + grad clip 1.0 + bf16。
- **Pass criteria**：训练日志无 inf/nan。

#### Task 5.6 — Risk R9（v2.1 新增）: Anchor 结构冷启动退化
- **风险描述**：原 ckpt 训练时无 anchor conditioning，引入 anchor embedding + 重定义 IL loss（从纯 noise 起点改为 anchored 起点）可能让 expert 起步阶段 PDMS 从 87.2 显著回退。
- **Action**：Phase 3.2 的渐进式 anchor 系数 $\lambda_a$ warmup（500 steps from 0→1）。监控 Phase 3 早期 navtest PDMS 是否下落超过 2.0。
- **Verification**：每 100 steps eval navtest PDMS（前 1000 steps 高频监控）。
- **Pass criteria**：训练全程 PDMS 不低于 85.5；最终回到 ≥ 86.0。

#### Task 5.7 — Risk R10（v2.1 新增）: Backbone 容量瓶颈
- **风险描述**：完全 frozen backbone 可能不足以让 anchor 结构充分发挥（特别是 backbone 没见过 anchor conditioning 信号）。如 Phase 3 PDMS 严格停滞在 84-85，说明 expert 单方面无法补偿。
- **Action**：Tier-1 兜底（默认不启用）：仅在 Phase 3 PDMS 卡在 < 85.5 持续 1k steps 时启动，对 backbone 添加 LoRA(r=8, α=16) 在 q_proj/v_proj。Tier-2 兜底：放开 Action Expert 与 backbone 之间的 cross-attention 层（在 Joint Attention 中识别这些层并解冻）。
- **Verification**：Tier-1 启用后 1k steps 内 navtest PDMS 是否爬出 plateau。
- **Pass criteria**：要么默认配置达标 ≥ 86.0；要么 Tier-1 兜底后达标。

---

### Phase 6：Mode Selector Stage 3（3-4 days）

#### Task 6.1 — 生成 Selector 训练数据
- **Action**：用 Stage 2-B 最终 ckpt 在 navtrain rollout 80 条/scene 轨迹 + PDM score。
- **Verification**：缓存 ~10GB；spot check 100 scenes。
- **Pass criteria**：数据完整。

#### Task 6.2 — Selector 训练
- **Action**：复用 `mode_selector.py`，20 epochs。
- **Verification**：top-1 accuracy improving。
- **Pass criteria**：top-1 acc 比 random 高 ≥ 30%。

#### Task 6.3 — Pipeline 集成
- **Action**：FM expert 生成 → selector 选 top-1 → 输出 trajectory。
- **Verification**：H100 推理 latency < 220ms。
- **Pass criteria**：latency + final navtest PDMS ≥ 90.0。

---

### Phase 7：评估与消融（3-5 days）

#### Task 7.1 — NAVSIM v1 完整评估
- **Action**：报告 NC, DAC, TTC, C, EP, PDMS。
- **Pass criteria**：完整 table。

#### Task 7.2 — Diversity & Top-K PDMS
- **Action**：每 scene 生成 20 条，报告 PDMS@1/5/10。
- **Pass criteria**：Top-1 > IL baseline；Top-10 显著提升。

#### Task 7.3 — 消融实验
- **Ablations**：(a) 不用 anchor；(b) 不用 Inter-Anchor truncation；(c) additive vs multiplicative noise；**(d) v2.1 新增**：Anchor 系数 $\lambda_a$ 不做 warmup 直接设 1，验证 R9 的实际影响。
- **Pass criteria**：每项至少差 0.5 PDMS。

#### Task 7.4（可选）— AR World Model Reward 实验（Phase 2 探索）
- **Action**：用 AR-WM rollouts 替代部分 PDM scorer。
- **Pass criteria**：训练速度提升 ≥ 1.5x，PDMS 不显著下降。

---

## 3. 风险与解决方案对照表（v2.1 最终）

| Risk | 风险描述 | v2.0 状态 | v2.1 状态 | 缓解方案 |
|------|---------|----------|----------|----------|
| R1 | Anchor σ 调参不当 | 活跃 | 活跃 | Phase 5.1 grid search |
| R2 | RL 梯度破坏 WM 表示 | 需 token head frozen | **完全消解**（WM 全 frozen + 不 forward） | Task 5.2 hash 校验 |
| R3 | Scorer 计算瓶颈 | 活跃 | 活跃 | Phase 5.3 async pool |
| R4 | navtrain 过拟合 | 活跃 | 活跃 | Phase 5.4 navtest + early stop |
| R5 | 6VA→2VA 适配 | 活跃 | **解决**（ckpt 已是 2VA） | — |
| R6 | log π 数值不稳 | 活跃 | 活跃 | Phase 5.5 min std + clip + bf16 |
| R7 | FM expert 代码缺失 | 活跃 | **解决**（ckpt 已发布即代码已 release） | — |
| R8 | Emu3 LoRA 适配 | 活跃 | **解决**（不再用 LoRA） | — |
| **R9（v2.1 新增）** | **Anchor 冷启动退化** | — | 活跃 | **Task 3.2 渐进 λ_a warmup + 高频 PDMS 监控** |
| **R10（v2.1 新增）** | **Backbone frozen 容量瓶颈** | — | 活跃 | **Task 5.7 两层兜底（Tier-1 LoRA / Tier-2 cross-attn 解冻）** |

**净变化**：3 项消解（R5/R7/R8）、2 项新增（R9/R10）、1 项简化（R2）。Risk 总数 8 → **6 项活跃**。

---

## 4. 文件结构（v2.1 调整）

```
drivevla_w0_grpo/
├── checkpoints/                    # v2.1 新增：分离后的 ckpt 存储
│   ├── backbone_emu3.pt            # frozen
│   ├── world_model_ar.pt           # frozen
│   ├── action_expert_init.pt       # 训练起点
│   └── stage2a_ref.pt              # IL 适配后 frozen reference
│
├── configs/
│   ├── stage2a_anchor_adapt.yaml   # IL 适配 (2k steps，含 λ_a warmup)
│   ├── stage2b_grpo.yaml           # GRPO main (10 epochs)
│   └── stage3_mode_selector.yaml   # Selector (20 epochs)
│
├── rl_modules/                     # ← 复用 DD-v2 代码（未变）
│   ├── anchor_kmeans.py
│   ├── multiplicative_noise.py
│   ├── intra_anchor_advantage.py
│   ├── inter_anchor_truncated.py
│   ├── pdm_reward_wrapper.py
│   ├── rollout_collector.py
│   ├── mode_selector.py
│   └── grpo_loss.py
│
├── action_expert/                  # ← 仅修改此目录的内容
│   ├── ckpt_loader.py              # Task 0.2 组件分离工具
│   ├── flow_matching_base.py       # 从 ckpt 加载的原始 FM expert wrapper
│   ├── anchor_embedding.py         # Task 2.2（新）
│   ├── anchored_flow_path.py       # Task 2.3（新）
│   ├── mixture_weight_head.py      # Task 2.4（新）
│   ├── ode_sampler.py              # Task 2.5（新）
│   └── log_pi.py                   # Task 2.6（新）
│
├── frozen/                         # v2.1 新增：frozen 组件 wrapper
│   ├── frozen_backbone.py          # Emu3 wrapper with hash assertion
│   └── frozen_wm.py                # AR-WM wrapper（仅作 module 容器，不 forward）
│
├── training/
│   ├── stage2a_anchor_adaptation.py
│   ├── stage2b_grpo.py
│   └── stage3_selector.py
│
├── evaluation/
│   ├── navtest_evaluator.py
│   ├── diversity_topk.py
│   └── ablation_runner.py
│
└── tests/
    ├── test_ckpt_split.py          # v2.1 新增：Task 0.2
    ├── test_frozen_hash.py         # v2.1 新增：Task 5.2
    ├── test_zero_anchor_compat.py  # v2.1 新增：Task 2.2 兼容性
    ├── test_anchored_fm.py
    ├── test_log_pi.py
    ├── test_grpo_loss.py
    └── test_e2e_smoke.py
```

---

## 5. 时间估算（v2.1 更新）

| Phase | Duration | Cumulative | 关键里程碑 |
|-------|----------|-----------|-----------|
| ~~Phase 0~~ | **跳过** | 0 | — |
| Task 0 (Ckpt 解析) | 0.5 day | 0.5 day | PDMS 87.2 复现 |
| Phase 1 | 3-5 days | 5.5 days | RL 模块单元测试全通 |
| Phase 2 | 5-7 days | 12.5 days | Anchored FM forward + zero-anchor 兼容性 |
| Phase 3 | 2-3 days | 15.5 days | 适配 navtest PDMS ≥ 86 |
| Phase 4 | 7-10 days | 25 days | GRPO PDMS ≥ 89 |
| Phase 5 | 并行 | (-) | 6 项 risk 验证全过 |
| Phase 6 | 3-4 days | 28 days | Selector 集成 PDMS ≥ 90 |
| Phase 7 | 3-5 days | 33 days | 评估 + ablation |

**总计：约 4-5 周**（v2.0 是 5-6 周，省下 Phase 0 的 1-2 天 + R5/R7/R8 解决节省的工作量 + Phase 3 由 4k 缩短为 2k steps）。

---

## 6. 关键超参对照表（v2.1 最终）

| 参数 | 值 | 备注 |
|------|---|------|
| **Phase 3 IL 适配（v2.1 简化）** | | |
| steps | **2k** | v2.0 是 4k；v2.1 因起点已是 PDMS 87.2 缩短 |
| anchor 系数 $\lambda_a$ warmup | 0→1 over 500 steps | **v2.1 新增 R9 缓解** |
| batch size | 512 | DD-v2 |
| Action Expert LR | 2e-4 | DD-v2 |
| **Backbone** | **完全 frozen** | v2.1 关键变更 |
| **World Model** | **完全 frozen + 不 forward** | v2.1 关键变更 |
| weight decay | 1e-4 | DD-v2 |
| **Phase 4 GRPO** | | |
| epochs | 10 | DD-v2 |
| N_anchor | 10 (8-12 grid) | DD-v2 |
| G per anchor | 8 | DD-v2 |
| T_trunc (训练) | 10 | |
| inference ODE 步 | 2 | DD-v2 |
| η (training) | 1 | DD-v2 |
| BC loss weight λ | 0.1 (overfit 时 0.2) | DD-v2 + R4 |
| discount γ | 0.8 | DD-v2 |
| min explore std σ_anchor | 0.04 (grid 0.02-0.16) | DD-v2 + R1 |
| min log-var std | 0.10 | R6 |
| gradient clip norm | 1.0 | R6 |
| precision | bf16 | R6 |
| **R10 兜底（默认不启用）** | | |
| Tier-1 LoRA | r=8, α=16 on backbone q/v_proj | 仅 Phase 3 PDMS 卡 < 85.5 时启用 |
| **Phase 6 Mode Selector** | | |
| epochs | 20 | DD-v2 |
| aug noise std | (0.1, 0.2) | DD-v2 |
| GTRS vocab pct | 1% | DD-v2 |

---

## 7. 验收标准（v2.1 最终）

- [ ] Task 0.3：navtest PDMS 87.2 ± 0.3 复现
- [ ] Phase 1：所有 RL 模块单元测试通过
- [ ] Phase 2：Anchored FM forward + zero-anchor 兼容性 + 5 个 unit tests
- [ ] Phase 3：navtest PDMS ≥ 86.0（允许相对 87.2 退化 ≤ 1.2）
- [ ] Phase 4：navtest PDMS ≥ 89.0（IL + 3）
- [ ] Phase 5：6 项活跃 risk 全部缓解验证
- [ ] Phase 6：完整 pipeline navtest PDMS ≥ 90.0
- [ ] Phase 7：完整 ablation table（含 R9 验证消融 d）+ Top-K PDMS 报告

---

## 8. v2.0 → v2.1 变更摘要

| 项 | v2.0 | v2.1 |
|---|---|---|
| 起点 | Stage 1 ckpt | **Stage 2 完整 ckpt PDMS 87.2** |
| Phase 0 | 1-2 天 | **跳过** |
| Backbone 微调 | LoRA(r=16) | **完全 frozen** |
| WM 处理 | Token head frozen | **整个 WM frozen + 不 forward + hash 校验** |
| Stage 2-A 性质 | 从零训练 | **adapt 已训 expert 到 anchored** |
| Stage 2-A 步数 | 4k | **2k** |
| Stage 2-A 关键技巧 | — | **anchor 系数 λ_a warmup 0→1** |
| Trainable params | ~537M | ~507M（无 LoRA） |
| Risk 数（活跃） | 8 | **6**（解 R5/R7/R8，新增 R9/R10） |
| 总时长 | 5-6 周 | **4-5 周** |
| Phase 1/6/7 | 不变 | 不变 |
| 方法学公式 | 不变 | 不变 |

**核心结论**：v2.1 通过依赖公开发布的 PDMS 87.2 ckpt，将工程范围缩小到"仅修改 Action Expert + 渐进引入 anchor 结构"，消除了 backbone 适配相关的所有风险。新增的 R9/R10 是 anchor 结构改造本身的内在风险，已通过 λ_a warmup（默认）+ 兜底解冻方案（备用）双重保护。
