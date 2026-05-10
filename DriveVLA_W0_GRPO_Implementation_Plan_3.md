# DriveVLA-W0 + DiffusionDrive v2 GRPO 详细实施计划

> **Phase 1 / NAVSIM 规模 / Flow Matching Action Expert**
> **Backbone: Emu3 8B（VLM，含 visual-token CE 作为 AR-WM 信号）+ Action Expert（小型 Emu3Model）**
> **Starting point: 本地 `pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2/` HuggingFace-style ckpt**
> Version: 2.2（基于 v2.1 的代码级 review 修订）

---

## 0. v2.2 修订说明

### 0.1 v2.2 修订动因

v2.1 完成后，对 `reference/Emu3/emu3/mllm/modeling_emu3.py`、`models/policy_head/`、`reference/DiffusionDriveV2/.../diffusiondrivev2_model_rl.py`、以及 ckpt 与训练脚本做了一次完整代码级 review，发现 v2.1 多处与实际代码不一致。v2.2 修正这些落差，并把"design 留给开发时再决定"的歧义点全部锁死。

### 0.2 实际代码事实（必须读）

1. **模型结构**（[reference/Emu3/emu3/mllm/modeling_emu3.py:1926-2003](reference/Emu3/emu3/mllm/modeling_emu3.py#L1926-L2003)）：
   - `Emu3Pi0` 内只有两大模块：`self.vlm`（`Emu3MoE`，含 `lm_head`）+ `self.action_expert`（`Emu3Model`），加上小头 `state_projector` / `action_projector` / `action_decoder` / `tau_emb`。
   - **没有独立的 `world_model.*` 模块**：所谓"AR World Model"在本 codebase 里 = `vlm.lm_head` 输出未来视觉 token logits + shifted CE（[modeling_emu3.py:2150-2206](reference/Emu3/emu3/mllm/modeling_emu3.py#L2150-L2206)）。"freeze WM" = 冻结 `vlm.lm_head` 并跳过 vlm_loss 计算。

2. **Action Expert 真实大小**（ckpt `config.json` 的 `action_config`）：`hidden_size=1024, intermediate_size=512, num_hidden_layers=32`。粗算 ≈ **150–250M**（不是 v2.1 文档反复出现的 500M）。

3. **当前 Flow Matching 时间约定**（[models/policy_head/noise_schedulers.py:44-62](models/policy_head/noise_schedulers.py#L44-L62)）：`z_t = (1-t)·data + t·noise`，**t=0 是 GT，t=1 是 noise**；target = `noise − data`；推理 ODE 走 t=1→0。这与 v2.1 §1.1 写的 "x_t = (1-t)·anchor + t·GT, t=0 是 anchor"**方向相反**。v2.2 锁定方案 A（保留当前代码约定，让 anchor 接管"noise 端"角色）。

4. **DD-v2 IL 项口径**（[diffusiondrivev2_model_rl.py:1105-1118](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1105-L1118)）：IL 是对 **GT 轨迹**的 L1，**没有** frozen reference policy。v2.1 引入的 "Stage 2-A reference checkpoint as KL anchor" 不在 DD-v2 中。v2.2 默认沿用 DD-v2 写法（vs GT L1），把 reference-policy KL 列为 ablation。

5. **DD-v2 探索噪声**：multiplicative `prev_sample = prev_mean·(1+ε_mul)`，`ε_mul = (ε_long, ε_lat)` 仅 2 个标量广播，min clip 0.04。v2.1 报告里的 "additive Gaussian per-element step noise" 与 paper §3.1 明确反对的写法相同，v2.2 改为 multiplicative。

6. **Mode Selector**：DD-v2 的 selector 用 BEV feature + map/agent queries。本仓库**没有 BEV encoder**，必须改为以 VLM hidden states 为 condition 的版本，不是"unchanged inherit"。

7. **训练入口**：`utils/train_pi0.py` 的 `init_fresh_expert=True` 分支（[train_pi0.py:142-148](utils/train_pi0.py#L142-L148)）会丢弃 ckpt 的 expert 权重重新初始化。GRPO 流程必须走 `from_pretrained` 分支（[train_pi0.py:149-163](utils/train_pi0.py#L149-L163)），即 `init_fresh_expert=False`。

8. **PDMS 87.2 起点（v2.2 第二次 review 更新）**：实际查阅 ckpt 内 `trainer_state.json` 显示该 ckpt 训了 **10000 steps / epoch 9.78** 且 loss 稳定下降到 ~0.005 量级，与"完整 navtrain 训练"画像一致——R11"baseline 可能是 mini 中间产物"概率被显著降低，但仍需 Task 0.3 实跑 navtest 复现 87.2 ± 0.5 才算锁定。

9. **Inference script anchor-free**：`inference/vla/inference_action_navsim_flow_matching_vava.py:171` 直接调 `model.sample_actions(...)` 不传任何 anchor 参数，确认现 87.2 ckpt 的推理路径**不带 anchor conditioning**。GRPO 后的推理必须**新写一份带 anchored ODE + Mode Selector 的 inference 脚本**（既有脚本不能复用）。

10. **Trajectory 归一化空间**（**重要**，第二次 review 新发现）：
   - 现有训练/推理的 `action` 张量在 `[-1, 1]` 归一化空间（`configs/normalizer_navsim_trainval/norm_stats.json`，按 q01/q99 mapping）。
   - `action_dim=3 = (x_long, y_lat, heading)`；mean(physical) ≈ (2.31 m, 0.012 m, 0.010 rad)；q01/q99 为 (-0.018, 6.2) / (-0.19, 0.24) / (-0.19, 0.18)。
   - **DD-v2 的 multiplicative `(1+ε_mul)` 在物理空间是有意义的**（典型 magnitude ~m），但**在归一化 [-1,1] 空间里**当某个 waypoint 接近 0（如 lateral / heading 在直行场景）时，`(1+ε_mul)·0 ≈ 0`，乘性噪声**几乎不起作用**（degenerate）。这是 v2.1/v2.2 §1.1 没有讨论的潜在 bug。
   - **v2.2 决议**：anchor 与 ε_mul 的施加都在**物理（米/弧度）空间**完成——即先把 anchor / GT 反归一化到物理量纲，应用 `(1+ε_mul)`，再归一化回 [-1, 1] 喂给 expert。详见 Task 2.3 v2.2 修订。

### 0.3 v2.2 → v2.1 主要变更

| 项 | v2.1 | v2.2 |
|---|---|---|
| ckpt prefix 切分 | `backbone.` / `world_model.` / `action_expert.`（不存在） | **`vlm.` / `action_expert.` / 小头**（实际） |
| Action Expert 参数量 | 500M | **~150–250M（待 Task 0.2 实测）** |
| Trainable 总量 | ~507M | **~150–250M + 新增 ~7M anchor/mixture heads** |
| Time 约定 | 仅 paper 公式（与代码反向）| **方案 A：保留代码约定，anchor 进入"noise 端"** |
| Anchor 注入位置 | "additive at first layer alongside time emb"（含糊） | **方案 (a)：作为额外 prefix token 与 state_token 并列** |
| cmd 与 anchor 关系 | 未定义 | **保留 cmd（兼容已训行为），anchor 作为细粒度补充** |
| Stage 2-B IL | vs Stage 2-A frozen reference (KL) | **vs GT L1（DD-v2 默认），reference-KL 转为 ablation** |
| 探索噪声 | additive Gaussian per-step | **multiplicative `(1+ε_mul)`，ε_mul 2 标量（DD-v2 默认）** |
| Mode Selector | 结构 unchanged | **重设计：以 VLM hidden states 为 condition，去掉 BEV/map/agent 路** |
| WM aux loss | optional β=0.01 残留 | **删除（与"WM 不 forward"矛盾）** |
| Phase 3 步数 | 固定 2k | **2k 起步 + early-stop 条件，可延至 4k** |
| Risk 数 | 6 项 | **8 项**（新增 R11 ckpt 真假基线、R12 cmd/anchor 冗余） |

### 0.4 IL baseline 处理

87.2 起点在 v2.2 中**不再被当成已验证锚点**。Task 0.3 的 PASS 条件改为"按 ckpt 文档复现到声明的数字 ± 0.5 PDMS"——具体目标值由 trainer_state.json 与 README 决定，而不是预设 87.2。如果 ckpt 实际达到的是 mini 数据上的 ~80 PDMS，Phase 3 的目标值要相应下调，并把 Phase 3 性质改为"完整 navtrain 上的 IL warm-up"（步数 ≥ 4k）。

---

## 1. 关键设计（v2.2 锁死）

### 1.1 Anchored Flow Matching 严格表述（**方案 A：与代码同向**）

设训练样本 $(\tau_{GT})$，K-Means 在 navtrain 上聚类得 $N_{anchor}=20$ 个 anchor $\{a^k\}_{k=1}^{20}$（与 DD-v2 默认 `ego_fut_mode` 一致；10/15/20 在 §Phase 5.1 ablation）。

**正样本 anchor**：$k^+ = \arg\min_k \|a^k - \tau_{GT}\|_2$（仅在 (x,y) 上算 L2，与 DD-v2 一致）。

**时间约定（与 [models/policy_head/noise_schedulers.py](models/policy_head/noise_schedulers.py) `add_noise` 一致）**：t ∈ [0,1]，**t=0 是 GT，t=1 是"anchor + 探索噪声"**；ODE 推理走 t=1 → 0（anchor → GT）。这与 v2.1 写法方向相反，v2.2 锁定本约定，不再变动。

**Anchored 直线路径**：
- 终点（noise 端）：$x_1^{k^+} = a^{k^+} \odot (1 + \epsilon_{mul})$，$\epsilon_{mul} = (\epsilon_{long}, \epsilon_{lat}) \sim \mathcal{N}(0, \sigma_{anchor}^2 I_2)$，仅 **2 个标量**广播到所有 waypoint（DD-v2 §3.1，preserves 平滑性，避免 jagged）。
- 路径：$x_t^{k^+} = (1-t) \cdot \tau_{GT} + t \cdot x_1^{k^+}$
- **目标向量场（直线路径下为常向量）**：$v^*(x_t^{k^+}, t) = x_1^{k^+} - \tau_{GT}$（注意符号与 v2.1 相反——这才是与现有代码 `MSE(noise − data, v_θ)` 一致的写法）
- **IL Loss**：$L_{FM\text{-}IL} = \mathbb{E}_t \left[ \| v_\phi(x_t^{k^+}, t, z, a^{k^+}) - (x_1^{k^+} - \tau_{GT}) \|^2 \right]$
- **仅正 anchor 反传**：$L_{FM\text{-}IL}$ 只在 $k^+$ 上计算梯度，负 anchor 不施加 reconstruction supervision（防止 collapse-to-mean，对应 DD-v2 paper Eq. 4 的 $y^k=1$ mask）。

**Mixture Weight Head**：
- 结构：MLP，输入 anchor-conditioned 池化特征 → 标量 logit $\hat s^k$
- $L_{BCE} = \sum_k \text{BCE}(\hat s^k, \mathbf{1}[k=k^+])$

**Stage 2-A 总 IL loss**：$L_{IL} = L_{FM\text{-}IL} + L_{BCE}$

### 1.2 GRPO 阶段（v2.2 锁死）

复用 DD-v2 GRPO，关键参数：
- 每个 anchor 采 G=8 条 trajectory（DD-v2 默认 G=10；v2.2 取 8 节省 scorer 工作量）
- Truncated denoising：训练 $T_{trunc}=10$ 步，推理 $T_{infer}=2$ 步
- Intra-anchor advantage（按 G 维做 mean/std）：$A^{k,i} = (r^{k,i} - \mu_k) / (\sigma_k + 10^{-4})$
- Inter-anchor truncation：$A^{k,i}_\text{trunc} = -1$ if collision/drivable_area 违规，else $\max(0, A^{k,i})$，并仅在"$r^{k,i}>r_{GT}-10^{-6}$"的样本上保留正 advantage（DD-v2 [model_rl.py:892](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L892)）
- 时序折扣 $\gamma=0.8$，按 denoising step 衰减（DD-v2 [model_rl.py:925-932](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L925-L932)）
- 探索噪声 multiplicative：每步 $x_{t+\Delta t} = x_t^{mean}(1+\epsilon^{step}_{mul})$，$\epsilon^{step}_{mul}\sim\mathcal{N}(0, \sigma_{step}^2 I_2)$，min clip 0.04
- Loss：REINFORCE `per_token_loss = -exp(logp - logp.detach()) * advantages`（DD-v2 [model_rl.py:1096](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1096)），按非零 token 平均
- 自适应 IL 权重：`il_weight = 0.1 if has_positive else 1.0`（DD-v2 [model_rl.py:1115-1117](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1115-L1117)）；IL 项是 **L1 vs GT trajectory 的 (x,y)**（DD-v2 [model_rl.py:1105-1112](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1105-L1112)），不引入 reference policy

### 1.3 Reward 严格定义（与 DD-v2 一致）

$$R(\tau) = \begin{cases} -1 & \text{if collision} \\ NC \cdot DAC \cdot \frac{5\,EP + 5\,TTC + 2\,C}{12} & \text{otherwise} \end{cases}$$

---

## 2. 顺序子任务（TDD-style）

### Task 0 — Checkpoint 解析与基线复现（1 day，扩充）

> 这是计划的实际起点。先核验起点 ckpt 的真实状态，再决定后续流程。

#### Task 0.1 — Ckpt 加载与 trainer_state 检视
- **Action**：本地 ckpt 在 `pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2/`（4 个 safetensor shard）。用 `safe_open` 列出所有 key，按前缀分桶；同时读 `trainer_state.json` / `training_args.bin` 确认：
  1. 训练数据规模（mini 还是完整 navtrain）
  2. 训练步数 / epochs / batch size
  3. `init_fresh_expert` / `freeze_vlm` / `action_loss_weight` 等关键 flag
  4. 最后一次 eval 的指标（如有）
- **Verification**：打印 prefix 直方图（预期主要是 `vlm.*` 与 `action_expert.*` + 小头），将 trainer_state 摘要写入 review 文件。
- **Pass criteria**：能明确说出"这个 ckpt 是在 X 数据/Y 步数上训练得到，eval 指标为 Z"。**这一步决定后续 Phase 3 目标值**。

#### Task 0.2 — 组件加载（按真实 prefix）
- **Action**：用 `Emu3Pi0.from_pretrained(...)` 直接加载（[utils/train_pi0.py:149-163](utils/train_pi0.py#L149-L163)），**不要走 `init_fresh_expert=True` 分支**。然后按真实结构标注 trainable / frozen：
  ```python
  # 实际 prefix 仅有：
  #   vlm.*                        → backbone（含 lm_head 即 "AR-WM"）
  #   action_expert.*              → 小型 Emu3Model
  #   state_projector.* / action_projector.* / action_decoder.* / tau_emb.*
  #
  # GRPO Stage 2-A/B trainable 分组：
  #   action_expert.*              → 全开
  #   action_projector.* / action_decoder.* / tau_emb.*  → 全开（低参，避免与 expert 失配）
  #   anchor_embedding.*           → 新增，全开
  #   mixture_weight_head.*        → 新增，全开
  #
  # Frozen：
  #   vlm.*  （含 lm_head；train_pi0.py 的 freeze_vlm 路径）
  #   state_projector.*  （保留对 cmd 的现有理解，不动）
  ```
- **Verification**：`model.from_pretrained` 报告 `missing_keys=[]`、`unexpected_keys=[]`（Task 0.2 阶段尚未引入新 anchor/mixture heads，所以应严格干净）；从 ckpt 加载完成后 `sum(p.numel() for p in model.action_expert.parameters())` 落在 **150–250M** 区间。
- **Pass criteria**：加载干净 + 实测 action_expert 参数量符合预期；同时打印 `vlm.*` 与各小头参数量并写入实验日志。

#### Task 0.3 — Baseline 复现
- **Action**：跑 navtest 推理（用既有 `inference/` pipeline），得到该 ckpt 的真实 PDMS。**目标值由 Task 0.1 的 trainer_state 决定**——如果 ckpt README 声明 87.2，复现 ± 0.5；如果是 mini 数据上的中间产物，记录实际值（可能远低于 87.2）。
- **Verification**：navtest PDMS 落在 ckpt 文档声明值 ± 0.5 内；forward 后 frozen 参数（vlm.*）hash 不变。
- **Pass criteria**：基线锁定 + frozen 参数 hash 校验通过。**记录这个数字 B0 作为后续 Phase 3/4 PDMS 目标的相对参照**：Phase 3 ≥ B0 − 1.2，Phase 4 ≥ B0 + 2.0。

---

### Phase 1：DiffusionDrive v2 RL 模块抽取（3-5 days）

> **此 Phase 完全不受任何变更影响**，DD-v2 RL 模块在轨迹空间工作，与 backbone/ckpt 形态解耦。

#### Task 1.1 — `anchor_kmeans.py`
- **Action**：从 DD-v2 复制 K-Means anchor 聚类逻辑，适配 NAVSIM 8 waypoints 格式。接口 `fit(trajectories_xy) → np.ndarray[N_anchor, N_f, 3]`：
  - 输入：navtrain GT 轨迹的 (x, y) 通道（**物理空间，米**），shape `(N_scenes, 8, 2)`。NAVSIM navtrain 实际有数万个场景（navtest ≈ 12k 样本作为对照），不是 1192——具体场景数读 `data/navsim/processed_data/scene_files/scene_filter/navtrain.yaml`。
  - K-Means 在 `(N_scenes, 16)`（flatten 8×2）上聚类。
  - 输出第 3 通道 (heading) 由 DD-v2 的 [`bezier_xyyaw`](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1123) 从 (x,y) 反推（与 DD-v2 一致），保证 heading 与 (x,y) 自洽。
- **Verification**：对全部 navtrain GT 聚类（cache 到 `cache/anchor_centers_N20.npy`），可视化 cluster 中心 + 各 cluster 50 条样本叠加图。
- **Pass criteria**：silhouette score > 0.3（DD-v2 navtrain anchor 实际数量级；0.4 偏严格）；可视化能区分直行 / 左转 / 右转 / 换道 / 停车意图。

#### Task 1.2 — `multiplicative_noise.py`
- **Action**：复用 DD-v2 的 $\epsilon_{mul} = (\epsilon_{long}, \epsilon_{lat})$ 2-标量乘性噪声。接口：
  ```python
  def apply_multiplicative_noise(
      traj_physical: torch.Tensor,   # (B, N_f, 3) in METERS/RADIANS, NOT [-1,1]
      sigma: float,                   # std of N(0, σ²)
      min_clip: float = 0.04,
  ) -> torch.Tensor:
      """Returns traj * (1 + eps_mul) on (x,y) channels only; heading unchanged."""
  ```
  **关键**：输入必须是物理空间张量（米/弧度）。如果调用方持有归一化张量，先 `denormalize → apply_noise → normalize` 三步走（v2.2 §0.2 #10）。
- **Verification**：unit test 覆盖 (i) σ=0 → 输出 == 输入；(ii) heading 通道严格不变；(iii) 加性 vs 乘性轨迹形状对比可视化（DD-v2 paper Fig. 3 的复现）。
- **Pass criteria**：3 项 unit test 通过；可视化无 jagged path。

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
- **Action**：复用 DD-v2 rollout 收集器骨架，返回的 `RolloutBatch` 字段（DD-v2 二段式 forward 所需最小集合）：
  ```
  RolloutBatch(
      anchors,         # (B, N_anchor, N_f, 3)             # 选用的 anchor 列表
      x_t_per_step,    # (B, K, T, N_f, 3)                 # 每步 noisy state（用于 Pass 2 重算 logπ）
      eps_step,        # (B, K, T, 2)                       # 每步乘性噪声 (long, lat)
      logp_old,        # (B, K, T)                          # Pass 1 时 detach 的 logπ
      rewards,         # (B, K)                             # PDM 总分
      sub_rewards,     # dict of (B, K)                     # NC/DAC/EP/TTC/C 子项
      advantages,      # (B, K, T)                          # truncated + discounted
      trajectories,    # (B, K, N_f, 3)                     # 每条 candidate 的最终 trajectory（K=N_anchor·G）
      gt_trajectory,   # (B, N_f, 3)                        # 训练时 broadcast 用
  )
  ```
  其中 K = N_anchor × G。
- **Verification**：feed 4 scenes，N_anchor=20, G=8, T=10 → K=160；shape 全部正确，logp_old 数值有限。
- **Pass criteria**：shape 校验 + logp finite + 内存占用合理（4 scenes × 160 × 10 × 8 × 3 × 4B ≈ 0.6 MB，可忽略）。

#### Task 1.7 — `mode_selector.py` 骨架（VLM-conditioned）
- **Action**：在 Phase 1 阶段先建空骨架（仅 `nn.Module` 接口、forward shape 校验），具体设计与训练放到 Phase 6 Task 6.1。本阶段不再尝试"复用 DD-v2 BEV 版本 selector"——本仓库无 BEV encoder，详见 §Phase 6。
- **Verification**：dummy `(trajectory, vlm_h)` 喂入返回正确 shape `(B, K)`。
- **Pass criteria**：骨架 forward 跑通即可，训练效果在 Phase 6 评。

---

### Phase 2：FM Action Expert Anchored 改造（5-7 days）

> **核心实施任务**——在已训练好的 FM expert 上添加 anchor 结构，**不动 backbone 和 WM**。

#### Task 2.1 — Action Expert 现有接口梳理
- **Action**：基于 [modeling_emu3.py:1926-2336](reference/Emu3/emu3/mllm/modeling_emu3.py#L1926-L2336)，把当前 Action Expert 的输入构造从代码中抽出来，确认改造点。当前每步 forward 的 action 端输入是：
  ```
  action_h = [state_token; action_projector(noisy_action, tau_emb)]
  # state_token = state_projector([pre_action.flatten(); cmd_one_hot])
  # action_projector: (noisy_action, tau_emb) → (B, action_frames, action_hidden)
  ```
  shared_layers 把 `(vlm_h, action_h)` 一起跑 32 层；action_decoder 在最后一层从 action token 部分解码出 vector field。
- **Verification**：写 dummy forward 把 expert 与 vlm decouple 出来跑一次（绕过 dataset），确认 shape 一致。
- **Pass criteria**：能在 unit test 中独立调用 expert 部分。

#### Task 2.2 — Anchor 注入：方案 (a) 额外 prefix token
- **决策**：v2.2 锁定 **方案 (a)**——把 anchor 投影成一个 token，作为 `state_token` 之外的第二个 prefix token：
  ```
  action_h = [state_token; anchor_token; action_projector(noisy_action, tau_emb)]
  ```
  原因：(i) 与 DD-v2 的 anchor query 思路对齐；(ii) 不污染 per-frame action_projector 的训练过的权重；(iii) attention mask 改动最小（仅扩 1 列）。
- **Action**：
  1. 新建 `models/policy_head/anchor_embedding.py`：
     ```python
     class AnchorEmbedding(nn.Module):
         def __init__(self, n_waypoints, action_dim, d_model):
             super().__init__()
             self.mlp = nn.Sequential(
                 nn.Linear(n_waypoints * action_dim, d_model),
                 nn.SiLU(),
                 nn.Linear(d_model, d_model),
             )
         def forward(self, anchor):                # (B, N_f, 3)
             return self.mlp(anchor.flatten(1))    # (B, d_model)
     ```
  2. 在 `Emu3Pi0.forward` / `sample_actions` 里把 `anchor_token = self.anchor_embedding(a_k)` unsqueeze(1) 与 `state_token_embedding` 拼接，再拼 action_projector 输出。
  3. **更新** `create_causal_style_attention_mask` 中的 `action_seq_len`：从 `1 + N_f` 改为 `2 + N_f`，并把新 prefix 列的可见性按 state_token 同样对待。
  4. **cmd 与 anchor 关系**：保留 cmd（不变 state_token）。anchor 作为细粒度多模意图补充。Phase 5 加 ablation "anchor + cmd 双注入 vs anchor only vs cmd only"以验证冗余度。
- **Verification**：
  - 不同 anchor → 不同 expert 输出（per-trajectory L2 diff > 1e-3）
  - 把 anchor_embedding 输出乘 0（zero-anchor）+ 不开 anchored path → expert 数值与原 ckpt forward 偏差 < 1e-3（保证 Phase 3 起步不退化）
- **Pass criteria**：上述两项 smoke test 通过。

#### Task 2.3 — Anchored Flow Path（与代码同向，方案 A，物理空间施加噪声）
- **Action**：实现 `models/policy_head/anchored_flow_path.py`，按 §1.1 锁定的方案 A：
  ```python
  # 输入：anchor 与 tau_GT 都是归一化 [-1,1] 张量（与现有训练 pipeline 一致）
  # 关键：multiplicative noise 必须在物理空间施加（v2.2 §0.2 #10）
  
  # 1) 反归一化 anchor 到物理空间（米/弧度）
  anchor_phys = denormalize(anchor, q01, q99)                 # (B, N_f, 3) meters/rad
  
  # 2) 物理空间施加 multiplicative noise（仅 x,y 通道）
  # 直接调用 Task 1.2 已交付的 apply_multiplicative_noise() 原语（返回 eps_xy 供 Task 2.6 log-π）
  from models.policy_head.multiplicative_noise import apply_multiplicative_noise
  x1_phys, eps_xy = apply_multiplicative_noise(anchor_phys, sigma=sigma_anchor, min_clip=0.0)
  # ε hard-clip（防止 (1+ε) 翻号）；注意 min_clip 是 σ floor，与 ε 上下界 eps_abs_clip 不同
  eps_xy = eps_xy.clamp(-0.5, 0.5)
  x1_phys = cat([anchor_phys[..., :2] * (1 + eps_xy), anchor_phys[..., 2:3]], -1)  # heading 不变
  
  # 3) 重新归一化回 [-1,1] 空间
  x1 = normalize(x1_phys, q01, q99)
  
  # 4) 直线路径（归一化空间，与 expert 输入一致）
  # x1 对应原 pure-noise 路径的 `noise = randn_like(action)`（t=1 端点，先验）
  x_t    = (1 - t) * tau_GT + t * x1
  target = x1 - tau_GT       # 与 (noise - action) 同向，复用现有 MSE 写法
  ```
  - **t=0 → x_t = τ_GT（数据）；t=1 → x_t = x1（anchor 邻域先验）**（与 [noise_schedulers.py](models/policy_head/noise_schedulers.py) 约定一致）。`x1` 替换了原 pure-noise 路径里 `noise = randn_like(action)` 的角色。
  - `q01/q99` 从 `configs/normalizer_navsim_trainval/norm_stats.json` 一次性加载到 `Emu3Pi0` 的 non-persistent buffer（`action_q01`/`action_q99`），避免每步 file IO，随 `.to(device, dtype)` 自动迁移。**注意：该 JSON 的顶层 data key 是 `"libero"`（历史命名遗留，数值实际是 NAVSIM 量纲）**，读取方式 `norm_cfg["norm_stats"]["libero"]`，与所有 inference 脚本一致，不重命名。
  - Task 2.3 **仅改 `Emu3Pi0.forward`**；`sample_actions` / `sample_actions_with_kv_cache` 的 anchor 初始化由 Task 2.5 stochastic ODE sampler 统一接管。
  - 同样的 denorm-perturb-renorm 模式也用于 Stage 2-B 每步采样器的 ε_step（Task 2.5）。
- **Verification**：
  - σ_anchor=0 时 `x_t = (1-t)·τ_GT + t·anchor_renorm`，与原始（无噪声）插值差距 < 1e-5；
  - t=0 时 `||x_t − τ_GT|| < 1e-6`；t=1 时 `||x_t − x1|| < 1e-6`；
  - heading 通道在 t=1 时严格等于 anchor 的 heading（即归一化空间也不变）；
  - Lateral 通道：σ_anchor=0.04，anchor lateral 物理值 0.05m → 加 ε_mul=0.04 后 → 物理 0.052m → 归一化变化量级 ~0.01（不再为 ~0）；
  - 梯度流：x_t / target 对 tau_GT 可微（backward 无 NaN/Inf）。
- **Pass criteria**：5 项 unit test 全过。

#### Task 2.4 — Mixture Weight Head
- **Action**：`models/policy_head/mixture_weight_head.py`：MLP（`Linear(h→h)→SiLU→Linear(h→1)`，零初始化最后一层），输入直接取 `final_action_hidden_for_decode[:, 1, :]`（anchor_token 那一位的单 token 隐状态，**无需额外池化**），输出 `(B,)` 标量 logit。零初始化保证 step 0 sigmoid=0.5（与 AnchorEmbedding 联合实现 87.2 ckpt bit-compat 起步）。
  - **调用位置**：在 `Emu3Pi0.forward` 中 `final_action_hidden_for_decode` 计算之后；`anchor=None` 走旁路（返回 `mixture_logit=None`）。返回经新 dataclass `Emu3Pi0Output(CausalLMOutputWithPast)` 的 `mixture_logit` 字段流出。BCE Loss 组装在 Task 3.3 trainer 外完成。
  - **多 anchor batching**：训练时 anchor 维度 N_anchor 由调用方 `repeat_interleave(N_anchor, dim=0)` 折入 batch；head 输出 `(B*N_anchor,)`，Trainer 外部 reshape 回 `(B, N_anchor)` 再算 BCE。Emu3Pi0 不感知 N_anchor。
  - ⚠️ **v2.2 修订**：`sample_actions` / `sample_actions_with_kv_cache` 在 Task 2.4 **不改**；推理路径下 anchor pick 由 Task 2.5 stochastic ODE sampler 统一接管。
- **Verification**：每个 anchor 独立 forward，logit shape `(B,)` 正确；`sigmoid(logit) ∈ (0, 1)`（不是 softmax：loss 是逐 anchor 独立 sigmoid + BCE，见 §1.1）；不同输入产生不同 logit（非零初始化后）；gradient flow 通过。
- **Pass criteria**：5 项 unit test 全过（shape / 零初始化 / sigmoid 范围 / 输入差异化 / 梯度流）。

#### Task 2.5 — Stochastic ODE Sampler（multiplicative，物理空间噪声）
- **决策**：v2.2 锁定 multiplicative 噪声（DD-v2 paper §3.1，反对 additive）；与 Task 2.3 同样在物理空间施加。
- **模块结构**：单步数学（`stochastic_euler_step`）放 `models/policy_head/stochastic_ode_sampler.py`（纯函数，isolation-testable）；多步循环放 `Emu3Pi0.sample_actions_stochastic` 方法，复用 `sample_actions` 的 forward chain。
- **Action**：实现两套 sampler：
  - **训练采样器**（T_trunc=10 步，stochastic，归一化空间走 ODE，物理空间加噪声）：
    ```
    # Euler step in normalized space
    z_{t-Δt}^mean_norm = z_t_norm + dt · v_φ(z_t_norm, t, anchor_token, vlm_h)
    # (dt = -1/T_trunc < 0)

    # denorm → multiplicative perturbation → renorm
    # q01/q99 来自 Emu3Pi0.action_q01 / action_q99 (已注册 buffer，同 Task 2.3)
    z_mean_phys      = denormalize(z_{t-Δt}^mean_norm, q01, q99)
    eps_xy           ∼ N(0, σ_step^2 I_2),  clip |·| ≤ 0.5,  σ_eff = max(σ_step, 0.04)
    z_next_phys      = cat([z_mean_phys[..., :2] * (1+eps_xy), z_mean_phys[..., 2:3]], -1)
                        # 仅 (x,y) 乘性噪声；heading 不变
    z_{t-Δt}_norm    = normalize(z_next_phys, q01, q99)
    ```
    采样器同时返回 `(z_{t-Δt}_norm, log_prob, z_{t-Δt}^mean_norm, eps_xy)`。
    多步 dict 键：`z_traj (T+1,B,N_F,3)`、`z_mean_traj (T,B,N_F,3)`、`log_prob_traj (T,B)`、`eps_step_traj (T,B,1,2)`。
  - **推理采样器**（T_infer=2 步，deterministic）：复用 `Emu3Pi0.sample_actions`，无需改动。
    推理路径入口为 `Emu3Pi0.sample_actions_anchored(anchor_K, ...)`：
    1. 探针 forward：把 K anchor `repeat_interleave(K, dim=0)` 折入 batch，调 `forward(anchor=anchor_BK, action=zeros, ...)` 拿 `mixture_logit (B*K,)` → view `(B, K)` → argmax → `anchor_pick (B, N_F, 3)`。
    2. 跑 `sample_actions(anchor=anchor_pick, num_inference_steps=T_infer=2)`。
    （Top-K / 加权平均留 Phase 4 优化。）
- **Verification**：
  - (i) 确定性重复：同 seed 多次跑 `stochastic_euler_step` 结果 bit-identical；
  - (ii) 蒙特卡洛收敛：50 次随机 seed 的 z_next 均值 ≈ z_mean（max diff < 1e-1）；
  - (iii) eps_xy shape `(B_eff, 1, 2)`（B_eff 含 K folded-in）；T 步堆叠后 reshape 回 `(B, K, T, 2)`；clip 上界 0.5 由 `apply_multiplicative_noise` + 显式 clamp 双重保护；
  - (iv) heading 通道不变：`z_next[..., 2] == z_mean[..., 2]`（atol 1e-5）。
- **Pass criteria**：7 项 unit test 全过（isolation，`tests/test_stochastic_ode_sampler.py`）。

#### Task 2.6 — Log Policy（multiplicative 对应）
- **v2.2 修订**：log π 写在 **z 上**（DD-v2 z 形式），而非 eps_step 上（eps 形式梯度恒零）：
  $$\log \pi_\theta(z_{next} | s, t) = \sum_{wp,\, c \in \{x,y\}} \left[ -\frac{(z_{next}^{\text{det}} - z_{mean})^2}{2\sigma_{lp}^2} - \log\sigma_{lp} - \tfrac{1}{2}\log 2\pi \right]$$
  其中 $z_{next}^{\text{det}} = z_{next}.\text{detach()}$（REINFORCE 梯度仅通过 $z_{mean}(\theta)$ 反传），$\sigma_{lp} = \max(\sigma_{step}, 0.10)$，仅 (x,y) 通道（N_F×2 = 16 维）求和。与 DD-v2 `DDIMScheduler_with_logprob` lines 668-676 数值等价（论文已验证 91.2 PDMS）。此 log-density 逻辑提取为 `_gaussian_log_prob_z` helper，由 rollout 与 Pass-2 recompute 共用，保证 IS ratio 数值 bit-identical。$\sigma_{lp}$ 下限 0.10 与 R6 (Task 5.5) 数值稳定要求同源，由 helper 统一 enforce，trainer 不需手动 clamp。
  - **rollout-pass**：`stochastic_euler_step` 内嵌调用 helper，同时返回 `log_prob`（复用 `test_stochastic_ode_sampler.py::test_log_prob_finite_100_batches` + `test_grad_flow_through_z_mean`）。
  - **Pass-2 recompute**：`recompute_log_prob(z_t_norm, velo_pred, z_next_phys_stored, dt, q01, q99, σ_step, σ_lp_min)` 纯函数，供 Task 4.3 GRPO trainer 在 stored z_next + 当前 θ 的 fresh velo_pred 上重算 log_prob，使 IS ratio `exp(lp_new - lp_old.detach())` 在 θ 更新后携带非零梯度。Emu3Pi0 多步循环 wrapper 留给 Task 4.3 编排。
- **Verification**：
  - (i) on-policy bit-identity：`recompute_log_prob(stored)` 与 rollout log_prob bit-identical (atol 1e-6)；
  - (ii) grad 仅经 velo_pred 反传，z_next_phys_stored.grad 为 None/zero；
  - (iii) IS ratio 在 θ 未变时 ≈ 1.0 (atol 1e-5)；
  - (iv) velo_pred 偏移 → log_prob 单调下降（Gaussian 密度惩罚）；
  - (v) σ_step=0 时 σ_lp 下限 0.10 仍生效（与 σ_step=0.10 结果 bit-identical）。
- **Pass criteria**：5 项新 unit test 全过（`tests/test_log_policy_recompute.py`）+ 复用 test_stochastic_ode_sampler.py test 6/7；全套 70 tests passed。

---

### Phase 3：IL 适配训练 Stage 2-A（2-4 days，含 early-stop 条件）

> v2.2：根据 Task 0.3 真实 baseline B0 决定步数与目标。
> 若 ckpt 是完整 navtrain 训练（PDMS ≈ 87.2 级别）→ 走"adapt 模式"，2k steps 起步；
> 若 ckpt 是 mini 数据中间产物（PDMS 远低于 87.2）→ 走"完整 IL warm-up 模式"，4k+ steps。

#### Task 3.1 — 训练脚本框架
- **Action**：新建 `utils/train_grpo_stage2a.py`（沿用 `train_pi0.py` 的 HF Trainer + dataset 框架）。**Trainable**：`action_expert.*` + `action_projector.*` + `action_decoder.*` + `tau_emb.*` + 新 `anchor_embedding.*` + `mixture_weight_head.*`。**Frozen**：`vlm.*`（含 lm_head，即 AR-WM 信号头）+ `state_projector.*` + tokenizer。
- **Verification**：单 step forward+backward 通过；trainable param count = Task 0.2 实测值 + ~7M 新 heads；`vlm.*` 的 `.grad` 为 None。
- **Pass criteria**：smoke test 通过 + 参数 hash 校验：vlm.* 训练前后不变。

#### Task 3.2 — 渐进式 Anchor 引入（R9 缓解）
- **Action**：anchor_embedding 的输出乘以 warmup 系数 $\lambda_a$：
  - 0 → 500 steps：$\lambda_a$ 线性 0→1（让 expert 起步沿用现 ckpt 行为）
  - 500+ steps：$\lambda_a = 1$
  - 同时 anchored path 的 σ_anchor 也做 0→0.04 线性 warmup
- **Verification**：每 100 steps eval navtest PDMS（前 1k steps 高频）。
- **Pass criteria**：训练全程 PDMS 不掉超过 max(2.0, B0 × 0.025)；最终回到 ≥ B0 − 1.2。

#### Task 3.3 — IL Loss 实现（vs GT，仅正 anchor）
- **Action**：组装
  $$L_{IL} = L_{FM\text{-}IL}(k^+\text{ only}) + L_{BCE}$$
  - $L_{FM\text{-}IL}$：仅在正 anchor $k^+$ 上反传 MSE（v2.2 §1.1）
  - $L_{BCE}$：所有 anchor 上的 BCE（正 anchor 标签 1，其他 0）
- **Verification**：初始 loss 数量级合理；前 100 steps 单调下降；用合成 batch（GT trajectory = 已知 anchor 中心 + 微噪声）验证 mixture_weight_head 能学出对应 anchor 高分。
- **Pass criteria**：loss 收敛趋势 + mixture_weight 合成验证通过。

#### Task 3.4 — 适配训练 + early-stop
- **Action**：默认 2k steps；如果 1k 步时 PDMS 还未回到 B0 − 2.0，触发"延长模式"再加 2k；如果 4k 步仍未达到 B0 − 1.2，挂起进入 R10 兜底（Tier-1 LoRA 解冻 vlm 的 q/v_proj）。
- **Verification**：每 500 steps eval navtest，绘制曲线 + 关键 metric（NC/DAC/EP/TTC）。
- **Pass criteria**：`final navtest PDMS ≥ B0 − 1.2`（v2.2 把绝对 86.0 改为相对 B0 的相对值）。

#### Task 3.5 — Stage 2-A ckpt 落盘（不再作 reference policy）
- **Action**：保存最终 ckpt 作为 Stage 2-B 的初始化点。**v2.2 取消"reference policy KL 锚"用法**——DD-v2 实际代码 IL 项是 vs GT trajectory L1（[diffusiondrivev2_model_rl.py:1105](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1105)），无需第二份 frozen 副本。reference-policy KL 留给 Phase 7 ablation（Task 7.5 新增）。
- **Verification**：ckpt 可被 Stage 2-B 脚本干净加载。
- **Pass criteria**：smoke 加载 + 单 step forward 一致。

---

### Phase 4：GRPO 训练 Stage 2-B（7-10 days）

> v2.2 实质变更（相对 v2.0/v2.1）：
> 1. trainable 集合 = `action_expert.*` + 小头 + 新 anchor/mixture heads（无 LoRA，除 R10 兜底）
> 2. 探索噪声 multiplicative，物理空间施加（§Task 2.3/2.5）
> 3. IL 项 = vs GT trajectory L1（DD-v2 默认），不是 vs frozen reference KL
> 4. PDMS 阈值改为相对 B0

#### Task 4.1 — Rollout 集成
- **Action**：把 Phase 1 的 `rollout_collector.py` 接入训练循环。
- **Verification**：单 batch 4 scenes，rollout 时间 < 5s。
- **Pass criteria**：throughput 满足要求。

#### Task 4.2 — 异步并行 Scorer
- **Action**：multiprocessing pool（16 workers）异步评估 PDM。
- **Verification**：throughput ≥ 200 trajectory/s；scorer 占比 < 30%。
- **Pass criteria**：满足两项指标。

#### Task 4.3 — GRPO Loss 集成（IL = vs GT L1，DD-v2 默认）
- **Action**：组装
  $$L = L_{RL} + \lambda_{IL} \cdot L_{IL,\text{vs-GT}}$$
  其中：
  - $L_{RL}$ = REINFORCE per-token loss，按非零 token 平均（DD-v2 [model_rl.py:1096-1102](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1096-L1102)）
  - $L_{IL,\text{vs-GT}}$ = 把 GT trajectory broadcast 到 `(B, N_anchor*G, T_decoder, ...)` 后做 (x,y) L1（DD-v2 [model_rl.py:1105-1112](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1105-L1112)）
  - $\lambda_{IL} = 0.1$ if `has_positive` else 1.0（DD-v2 自适应权重）
- **Verification**：loss 各项数值平衡；gradient norm < 1.0；初始几步 RL 项与 IL 项数量级在 1:10 ~ 10:1 之间。
- **Pass criteria**：数值合理 + 梯度不爆炸。

#### Task 4.4 — 1 Epoch Smoke Test
- **Action**：1 epoch GRPO 训练。
- **Verification**：navtest PDMS ≥ B0 − 1.2（与 Phase 3 一致或更高）。
- **Pass criteria**：稳定性通过。

#### Task 4.5 — 完整 10 Epochs GRPO
- **Action**：完整训练 + 每 epoch navtest + early stop（patience=2）。
- **Verification**：曲线监控 + best ckpt 保存。
- **Pass criteria**：`final navtest PDMS ≥ B0 + 2.0`（v2.2 相对目标，复制 DD-v2 IL→GRPO 提升幅度，约 +3 PDMS 但留 1 PDMS 作 Mode Selector 之前的余量）。

---

### Phase 5：风险缓解（与 Phase 4 并行执行）

#### Task 5.1 — Risk R1: Anchor σ 调参
- **Action**：grid search $\sigma_{anchor} \in \{0.02, 0.04, 0.08, 0.16\}$，每个 σ 跑 2 epochs。
- **Pass criteria**：选最佳 σ 进入完整训练。

#### Task 5.2 — Risk R2: vlm.* 表示保护
- **Action**：v2.2 下 `vlm.*`（含 lm_head）100% frozen，且训练 forward 中**跳过 lm_head 与 vlm_loss 计算**（避免无谓显存）。具体做法：
  1. `model.freeze_vlm()`（[utils/train_pi0.py:128-135](utils/train_pi0.py#L128-L135)）锁住 `requires_grad`
  2. 训练入口在 `Emu3Pi0.forward` 外加一层 wrapper，把 `labels` 传 None 或在 wrapper 中显式跳过 lm_head 调用（[modeling_emu3.py:2151-2206](reference/Emu3/emu3/mllm/modeling_emu3.py#L2151-L2206) 的整段在 `labels is not None` 时才执行）
  3. 训练循环每 N steps assert `hash(vlm.state_dict())` 不变
- **Verification**：训练前后 `vlm.*` 参数 hash 完全一致；显存占用对比"freeze 但仍 forward lm_head"的 baseline 减少。
- **Pass criteria**：hash 校验通过 + R2 自动消解。

#### Task 5.3 — Risk R3: Scorer 瓶颈
- **Action**：详见 Task 4.2。16-worker async pool + cache。
- **Pass criteria**：scorer 占比 < 30%。

#### Task 5.4 — Risk R4: navtrain 过拟合
- **Action**：每 epoch navtest 验证 + early stop（patience=2）。**v2.2 修订**：λ_IL 已是自适应（0.1 / 1.0），不再额外手调到 0.2；如确认过拟合，可改在自适应基础上把"has_positive 时的 0.1"乘 1.5–2.0 倍作 grid。
- **Pass criteria**：navtrain-navtest gap < 3 PDMS。

#### Task 5.5 — Risk R6: log-likelihood 数值不稳
- **Action**：log π 的 σ 下限 ≥ 0.10（见 §Task 2.6 `_gaussian_log_prob_z` helper，rollout 与 recompute 路径共同 enforce，trainer 不需手动 clamp）；ε_step 采样 std 下限 ≥ 0.04 + clip |·| ≤ 0.5；grad clip 1.0；bf16 训练。
- **Pass criteria**：训练日志无 inf/nan；前 100 steps grad norm < 1.0。

#### Task 5.6 — Risk R9: Anchor 结构冷启动退化
- **风险描述**：原 ckpt 训练时无 anchor conditioning，引入 anchor embedding + 重定义 FM 路径（从纯 noise 起点改为 anchored 起点）可能让 expert 起步阶段 PDMS 从 B0 显著回退。
- **Action**：Phase 3.2 的渐进式 anchor 系数 $\lambda_a$ warmup（500 steps from 0→1）+ σ_anchor warmup（同 500 steps from 0→0.04）。监控 Phase 3 早期 navtest PDMS 是否下落超过 max(2.0, B0×0.025)。
- **Verification**：每 100 steps eval navtest PDMS（前 1000 steps 高频监控）。
- **Pass criteria**：训练全程 PDMS ≥ B0 − 2.0；最终回到 ≥ B0 − 1.2。

#### Task 5.7 — Risk R10: Backbone 容量瓶颈
- **风险描述**：完全 frozen backbone 可能不足以让 anchor 结构充分发挥。
- **Action**：Tier-1 兜底（默认不启用）：仅在 Phase 3 PDMS 卡在 < B0 − 2.0 持续 1k steps 时启动，对 vlm 的 q_proj/v_proj 添加 LoRA(r=8, α=16)。**实际 LoRA 参数量**（GQA 修正）：
  - 配置（ckpt config.json）：`hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, head_dim=128`，故 q_proj 是 4096→4096，v_proj 是 4096→1024（GQA reduced）。
  - 每层 LoRA 参数 = (4096+4096)·8 + (4096+1024)·8 = 65536 + 40960 ≈ 0.106M
  - 32 层共 ≈ **3.4M**（v2.1 写的 32M、v2.2 第一版写的 8M 都偏大）。
  
  Tier-2 兜底：识别 `Emu3Pi0SharedLayer` 中负责 action→vlm K/V 的投影并仅解冻这部分。
- **Verification**：Tier-1 启用后 1k steps 内 navtest PDMS 是否爬出 plateau。
- **Pass criteria**：要么默认配置达标 ≥ B0 − 1.2；要么 Tier-1 兜底后达标。

#### Task 5.8 — Risk R11（v2.2 新增）: ckpt 真实 baseline 不达 87.2
- **风险描述**：仓库内现存的 mini 训练脚本（`train_navsim_flow_matching_mini.sh`）会让人误以为 ckpt 是 mini 中间产物。**第二次 review 后查 trainer_state.json 已确认 ckpt 为 10000 steps / epoch 9.78 完整训练，loss 收敛到 ~0.005**——R11 概率显著降低，但 PDMS 87.2 仍未独立验证（loss 收敛 ≠ PDMS 锁定，因为评测指标与训练 loss 不直接挂钩）。
- **Action**：Task 0.3 中以实测 B0 替代 87.2 假设；后续所有 PDMS 阈值改为相对 B0。如果 B0 < 80，要么换 ckpt（外部 release 或重训），要么把 Phase 3 改为完整 IL warm-up（5k+ steps，bs=512）。
- **Verification**：B0 测出后写入实验日志，所有阈值脚本化。
- **Pass criteria**：相对阈值方案可复现。

#### Task 5.9 — Risk R12（v2.2 新增）: cmd 与 anchor 信息冗余
- **风险描述**：现有 `state_token` 已注入 4 维 cmd one-hot，K-Means anchor 也是驾驶意图聚类，可能冗余甚至冲突（如 cmd=直行 + anchor=右转）。
- **Action**：默认保留两者并立。Phase 5.9.1 ablation：(i) anchor + cmd 共存（默认）；(ii) 训练时把 anchor 选取条件化在 cmd 上（同 cmd 的 trajectory 才进入 K-Means）；(iii) 移除 cmd 仅留 anchor。
- **Verification**：3 项 ablation 各跑 2 epochs。
- **Pass criteria**：选 PDMS 最高的方案进入主训练（默认假设是 (i)）。

---

### Phase 6：Mode Selector Stage 3（5-6 days，重设计）

> **v2.2 关键变更**：DD-v2 的 selector 结构（BEV cross-attn + map/agent queries + MLP）依赖一套外部 perception encoder，本仓库**没有 BEV 编码器**。Phase 6 改为基于 VLM hidden states 的轻量 selector。

#### Task 6.1 — VLM-conditioned Selector 设计
- **Action**：`models/policy_head/mode_selector.py`：
  ```
  # 输入：每条候选 trajectory τ̂^k (B, N_f, 3) + 该场景的 vlm_h (B, S, H_vlm)
  # 1. trajectory MLP encoder → q_traj (B, K, d_sel)
  # 2. cross-attn(q_traj, K=V=vlm_h)  → 单层 cross-attention（少头，d=256）
  # 3. self-attn over K candidates    → 让 candidates 互相比较
  # 4. 标量 score MLP                 → ŝ^k
  ```
  Loss：BCE（对正 mode）+ Margin-Rank（top-1 vs others），与 DD-v2 §3.2 同形。
- **Verification**：合成 (trajectory, score) 数据集训练 100 steps；单元测试通过。
- **Pass criteria**：smoke train + 单元测试通过。

#### Task 6.2 — 生成 Selector 训练数据
- **Action**：用 Stage 2-B 最终 ckpt 在 navtrain rollout `N_anchor × G = 20 × 8 = 160` 条/scene + PDM score（ckpt 推理 ODE 取 T_infer=2）。
- **Verification**：缓存 ~15GB；spot check 100 scenes 的 trajectory 是否在合理范围（坐标量级、heading 范围）。
- **Pass criteria**：数据完整 + spot check 通过。

#### Task 6.3 — Selector 训练
- **Action**：20 epochs，trajectory 增广 multiplicative noise std ∈ [0.1, 0.2]；1% supplementary trajectories 取自 GTRS 词表（如可用，否则跳过）。
- **Verification**：top-1 accuracy curve 单调上升。
- **Pass criteria**：top-1 acc 比 random 高 ≥ 30%。

#### Task 6.4 — 新 Anchored Inference 脚本 + Pipeline 集成
- **背景**：现有 `inference/vla/inference_action_navsim_flow_matching_vava.py:171` 直接调 `model.sample_actions(...)`，**完全 anchor-free**（与 GRPO 后的 anchored 推理路径不兼容）。GRPO 推理必须新写脚本。
- **Action**：
  1. 新建 `inference/vla/inference_action_navsim_anchored_grpo.py`：
     - 加载 anchor library + Mode Selector ckpt
     - 对每个 scene：跑 1 次 VLM forward（缓存 KV）→ 对每个 anchor 跑 anchored 2-step ODE（anchor 维 batch 化）→ 得到 N_anchor × G = 160 条 candidate（推理时可降到 G=1, K=20）→ Mode Selector 选 top-1
  2. 新建 `inference/vla/infer_navsim_grpo.sh`，相对 `infer_navsim_flow_matching_PDMS_87.2.sh` 增加 `VLA_ANCHOR_CLUSTER_PATH` 与 `VLA_SELECTOR_CKPT` 两个 env 变量
- **Verification**：H100 推理 latency < 220ms（K=20 时；K=160 不强求实时）；final navtest PDMS。
- **Pass criteria**：latency 达标（K=20 path）+ final navtest PDMS ≥ B0 + 3.0。

---

### Phase 7：评估与消融（3-5 days）

#### Task 7.1 — NAVSIM v1 完整评估
- **Action**：报告 NC, DAC, TTC, C, EP, PDMS。
- **Pass criteria**：完整 table。

#### Task 7.2 — Diversity & Top-K PDMS
- **Action**：每 scene 生成 20 条，报告 PDMS@1/5/10。
- **Pass criteria**：Top-1 > IL baseline；Top-10 显著提升。

#### Task 7.3 — 消融实验
- **Ablations**：
  - (a) 不用 anchor（退化为标准 FM + GRPO）
  - (b) 不用 Inter-Anchor truncation（保留负 advantage）
  - (c) additive vs multiplicative noise（v2.2 默认 multiplicative）
  - (d) Anchor 系数 $\lambda_a$ warmup 关掉直接设 1（验证 R9）
  - (e) **v2.2 新增**：cmd / anchor 注入消融（R12 三组）
  - (f) **v2.2 新增**：N_anchor ∈ {10, 15, 20} grid
  - (g) **v2.2 第二次 review 新增**：multiplicative 噪声施加空间——物理空间（默认）vs 归一化空间，验证 §0.2 #10 决议是否最优
- **Pass criteria**：每项至少差 0.5 PDMS。

#### Task 7.4（可选）— AR World Model Reward 实验（Phase 2 探索）
- **Action**：用 AR-WM rollouts（即 vlm.lm_head 上的 visual-token 自回归）替代部分 PDM scorer。
- **Pass criteria**：训练速度提升 ≥ 1.5x，PDMS 不显著下降。

#### Task 7.5（v2.2 新增）— Reference-Policy KL Regularization 消融
- **依赖**：Task 3.5 默认不再保存 frozen reference ckpt（v2.2 决议）。本 ablation 触发时，**临时**做：
  1. 在 Stage 2-A 训练结束时多保存一份 `stage2a_frozen_ref.pt`（独立于 Stage 2-B 的初始化 ckpt，仅在此 ablation 用）
  2. Stage 2-B 训练时同时加载 trainable model 与 frozen ref，每 step 各跑一次前向（额外 ~1.5x 显存）
- **Action**：把 §Task 4.3 的 IL 项从 "vs GT L1" 替换为 "vs Stage 2-A frozen reference 的 KL"，固定 KL 权重 0.05/0.1/0.2 各跑 2 epochs。KL 形式：在每步 ε_step 的 Gaussian 上算 `KL(N(0, σ_θ²) || N(0, σ_ref²))`（多元 Gaussian KL closed form），或者直接对采样 trajectory 算 L2 距离作 surrogate。
- **Pass criteria**：报告对比 PDMS / 训练稳定性。判断"是否值得引入 reference policy"——若 KL 版本能 ≥ vs-GT-L1 + 0.5 PDMS 且 grad norm 更稳定，则未来主路径可切换。

---

## 3. 风险与解决方案对照表（v2.2 最终）

| Risk | 风险描述 | v2.1 状态 | v2.2 状态 | 缓解方案 |
|------|---------|----------|----------|----------|
| R1 | Anchor σ 调参不当 | 活跃 | 活跃 | Phase 5.1 grid search |
| R2 | RL 梯度破坏 WM 表示 | 完全消解 | 完全消解（vlm.* 全 frozen，含 lm_head） | Task 5.2 hash 校验 |
| R3 | Scorer 计算瓶颈 | 活跃 | 活跃 | Phase 5.3 async pool（16 workers） |
| R4 | navtrain 过拟合 | 活跃 | 活跃 | Phase 5.4 navtest + early stop |
| R5 | 6VA→2VA 适配 | 解决 | 解决 | — |
| R6 | log π 数值不稳 | 活跃 | 活跃（multiplicative 形式） | Phase 5.5 min ε_step std=0.04, log-pi std=0.10, grad clip 1.0, bf16 |
| R7 | FM expert 代码缺失 | 解决 | 解决 | — |
| R8 | Emu3 LoRA 适配 | 解决（默认不用） | 解决（仅 R10 兜底用） | — |
| R9 | Anchor 冷启动退化 | 活跃 | 活跃 | Task 3.2 渐进 λ_a warmup + 高频 PDMS 监控 |
| R10 | Backbone frozen 容量瓶颈 | 活跃 | 活跃 | Task 5.7 两层兜底（Tier-1 LoRA r=8 ≈8M / Tier-2 SharedLayer 解冻） |
| **R11（v2.2 新增）** | **ckpt 真实 baseline 不达 87.2** | — | 活跃 | **Task 5.8 实测 B0 + 相对阈值方案 + 必要时延长 Phase 3** |
| **R12（v2.2 新增）** | **cmd 与 anchor 信息冗余** | — | 活跃 | **Task 5.9 三组 ablation 决定主路径** |

**净变化（vs v2.1）**：v2.1 的 6 项活跃保持不变，R2 描述更新为 vlm.* 全 frozen，新增 R11（ckpt 不确定性）+ R12（cmd vs anchor）。Risk 总数 = **8 项活跃**。

---

## 4. 文件结构（v2.2 与既有仓库对齐）

> v2.1 的"新建顶层 `drivevla_w0_grpo/`" 与现有仓库（`models/`, `utils/`, `scripts/`, `inference/`, `reference/`, `configs/`）脱节。v2.2 改为完全沿用既有目录命名。

```
DriveVLA-W0/                                    # 仓库根（已存在）
├── pretrained_models/
│   └── Emu3_Flow_Matching_Action_Expert_PDMS_87.2/   # 起点 ckpt（已存在）
│
├── configs/                                    # 已存在
│   ├── grpo/                                   # v2.2 新增子目录
│   │   ├── stage2a_anchor_adapt.yaml
│   │   ├── stage2b_grpo.yaml
│   │   └── stage3_mode_selector.yaml
│   └── ...                                     # 其他既有
│
├── models/policy_head/                         # 已存在
│   ├── flow_matching.py                        # 已存在（保留作 reference 实现）
│   ├── noise_schedulers.py                     # 已存在；v2.2 扩展 add_noise 支持 anchored 路径
│   ├── anchor_kmeans.py                        # v2.2 新增（Task 1.1）
│   ├── multiplicative_noise.py                 # v2.2 新增（Task 1.2）
│   ├── anchor_embedding.py                     # v2.2 新增（Task 2.2）
│   ├── anchored_flow_path.py                   # v2.2 新增（Task 2.3）
│   ├── mixture_weight_head.py                  # v2.2 新增（Task 2.4）
│   ├── stochastic_ode_sampler.py               # v2.2 新增（Task 2.5/2.6）
│   └── mode_selector.py                        # v2.2 新增（Task 6.1，VLM-conditioned）
│
├── utils/                                      # 已存在
│   ├── train_pi0.py                            # 已存在（Stage 2 现行 IL）
│   ├── train_grpo_stage2a.py                   # v2.2 新增（Task 3.1）
│   ├── train_grpo_stage2b.py                   # v2.2 新增（Phase 4）
│   ├── train_grpo_stage3.py                    # v2.2 新增（Task 6.3）
│   └── rl_modules/                             # v2.2 新增子目录
│       ├── intra_anchor_advantage.py
│       ├── inter_anchor_truncated.py
│       ├── pdm_reward_wrapper.py
│       ├── rollout_collector.py
│       └── grpo_loss.py
│
├── scripts/scripts_train/                      # 已存在
│   ├── train_navsim_grpo_stage2a.sh            # v2.2 新增
│   ├── train_navsim_grpo_stage2b.sh            # v2.2 新增
│   └── train_navsim_grpo_stage3.sh             # v2.2 新增
│
├── inference/                                  # 已存在；扩展 mode_selector 集成
│
├── reference/                                  # 已存在；只读引用
│   ├── DiffusionDriveV2/
│   └── Emu3/
│
└── tests/                                      # v2.2 新增（仓库目前无 tests/）
    ├── test_ckpt_load_real_prefix.py           # Task 0.2
    ├── test_frozen_vlm_hash.py                 # Task 5.2
    ├── test_zero_anchor_compat.py              # Task 2.2
    ├── test_anchored_flow_path.py              # Task 2.3
    ├── test_stochastic_sampler_logpi.py        # Task 2.5/2.6
    ├── test_grpo_loss.py
    └── test_e2e_smoke.py
```

---

## 5. 时间估算（v2.2 更新）

| Phase | Duration | Cumulative | 关键里程碑 |
|-------|----------|-----------|-----------|
| Task 0 (Ckpt 解析 + B0 复现) | 1 day | 1 day | trainer_state 摘要 + B0 锁定 |
| Phase 1 | 3-5 days | 6 days | RL 模块单元测试全通；mode_selector 仅出骨架 |
| Phase 2 | 5-7 days | 13 days | Anchored FM forward + zero-anchor 兼容性 + 物理空间噪声 unit tests |
| Phase 3 | 2-4 days | 17 days | 适配 navtest PDMS ≥ B0 − 1.2（含 early-stop 延长机制） |
| Phase 4 | 7-10 days | 27 days | GRPO PDMS ≥ B0 + 2.0 |
| Phase 5 | 并行 | (-) | 8 项 risk 验证全过 |
| Phase 6 | 5-6 days | 33 days | VLM-conditioned Selector 训完 + 新 anchored inference 脚本 + PDMS ≥ B0 + 3.0 |
| Phase 7 | 3-5 days | 38 days | 评估 + 6 项 ablation + reference-KL ablation |

**总计：约 5-6 周**（v2.2 比 v2.1 估的 4-5 周多 1 周——主要是 Phase 6 selector 重设计、Phase 7 ablation 增多、Task 0.3 真实 baseline 复现的开销）。

---

## 6. 关键超参对照表（v2.2 最终）

| 参数 | 值 | 备注 |
|------|---|------|
| **Phase 3 IL 适配** | | |
| steps | 2k 起步 + early-stop 延至 4k | 视 Task 0.3 实测 B0 决定 |
| anchor 系数 $\lambda_a$ warmup | 0→1 over 500 steps | R9 缓解 |
| anchor σ_anchor warmup | 0→0.04 over 500 steps | v2.2 新增；与 λ_a 同步 |
| batch size | 512 | DD-v2 |
| Action Expert LR | 2e-4 | DD-v2 |
| 新 heads LR multiplier | 1.0 | 与 expert 同步 |
| `vlm.*` | 完全 frozen（含 lm_head） | v2.2 关键 |
| `state_projector.*` | 完全 frozen | v2.2 决策（保留 cmd 已学行为） |
| weight decay | 1e-4 | DD-v2 |
| **Phase 4 GRPO** | | |
| epochs | 10 | DD-v2 |
| N_anchor | **20**（10/15/20 grid） | v2.2 与 DD-v2 默认对齐 |
| G per anchor | 8 | DD-v2 默认 10 → v2.2 取 8 节省 scorer |
| T_trunc (训练 denoising 步数) | 10 | DD-v2 |
| T_infer (推理 ODE 步数) | 2 | DD-v2 |
| 探索噪声形式 | **multiplicative**：`(1+ε_mul)`，ε_mul 仅 (long, lat) 2 标量 | v2.2 锁定（DD-v2 §3.1） |
| σ_anchor (Stage 2-A noise) | 0.04（grid 0.02-0.16） | DD-v2 + R1 |
| σ_step (Stage 2-B per-step ε) | 0.04（min clip） | R6 |
| log-π σ min clip | 0.10 | R6 |
| BC loss weight λ_IL | adaptive: 0.1 if has_positive else 1.0 | DD-v2 [model_rl.py:1115](reference/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py#L1115) |
| IL 项形式 | **vs GT trajectory L1 (x,y)** | v2.2 锁定（DD-v2 默认） |
| discount γ | 0.8 | DD-v2 |
| gradient clip norm | 1.0 | R6 |
| precision | bf16 | R6 |
| Scorer worker count | 16（ProcessPoolExecutor） | DD-v2 |
| **R10 兜底（默认不启用）** | | |
| Tier-1 LoRA | r=8, α=16 on `vlm.*.q_proj/v_proj` ≈ **8M** params | 仅 Phase 3 PDMS 卡在 < B0 − 2.0 持续 1k steps 时启用 |
| **Phase 6 Mode Selector（VLM-conditioned，v2.2 重设计）** | | |
| epochs | 20 | DD-v2 |
| Selector d_model | 256 | v2.2 新增 |
| Selector cross-attn 层 | 1 | 输入 vlm_h，q = trajectory 投影 |
| Selector self-attn 层 | 1 | candidates 互比 |
| aug noise std | (0.1, 0.2) | DD-v2 |
| GTRS vocab pct | 1%（如可用） | DD-v2 |

---

## 7. 验收标准（v2.2 最终）

> 所有 PDMS 阈值改为相对 Task 0.3 实测的 B0，避免对 87.2 的硬绑定。

- [ ] Task 0.1-0.3：trainer_state 摘要 + ckpt 干净加载（实际 prefix）+ navtest 复现到 ckpt 文档声明值 ± 0.5（B0 锁定）+ 物理 vs 归一化空间转换函数 unit test
- [ ] Phase 1：所有 RL 模块单元测试通过
- [ ] Phase 2：Anchored FM forward + zero-anchor 兼容性 + 6 个 unit tests（anchor_emb / anchored_path / mixture_head / sampler / log_pi / e2e_smoke）
- [ ] Phase 3：navtest PDMS ≥ B0 − 1.2（允许 anchor 结构引入小幅退化）
- [ ] Phase 4：navtest PDMS ≥ B0 + 2.0（复制 DD-v2 IL→GRPO 改善幅度）
- [ ] Phase 5：8 项活跃 risk 全部缓解验证（含 R11 baseline 真实性、R12 cmd/anchor 关系）
- [ ] Phase 6：完整 pipeline navtest PDMS ≥ B0 + 3.0
- [ ] Phase 7：完整 ablation table（含 v2.2 新增 (e) cmd/anchor、(f) N_anchor grid、Task 7.5 reference-policy KL）+ Top-K PDMS 报告

---

## 8. v2.1 → v2.2 变更摘要

| 项 | v2.1 | v2.2 |
|---|---|---|
| ckpt prefix 假设 | `backbone./world_model./action_expert.` | **`vlm./action_expert./` 小头**（实际） |
| Action Expert 大小 | 500M | **~150–250M（待 Task 0.2 实测）** |
| Trainable 总量 | ~507M | **~150–250M + ~7M 新 heads** |
| AR World Model 实体 | "独立模块"（不存在） | **= `vlm.lm_head` 上的 visual-token CE** |
| Time 约定 | x_t = (1-t)·anchor + t·GT（与代码反向） | **保留代码约定：x_t = (1-t)·GT + t·(anchor+ε_mul)，target = (anchor+ε_mul) − GT** |
| Anchor 注入 | 含糊"additive at first layer" | **方案 (a)：第二个 prefix token 与 state_token 并列** |
| 探索噪声 | additive Gaussian per-step | **multiplicative `(1+ε_mul)`，ε_mul 2 标量** |
| Stage 2-B IL 项 | vs Stage 2-A frozen reference (KL) | **vs GT trajectory L1（DD-v2 默认）；KL 转 ablation** |
| Mode Selector | "structurally unchanged from DD-v2" | **重设计：VLM-conditioned，去掉 BEV/map/agent** |
| Phase 3 步数 | 固定 2k | **2k 起步 + early-stop 延至 4k** |
| baseline 假设 | PDMS 87.2 硬锁 | **以 Task 0.3 实测 B0 为相对参照** |
| Risk 数（活跃） | 6 | **8（新增 R11 baseline 真实性、R12 cmd/anchor 冗余）** |
| 文件结构 | 顶层新建 `drivevla_w0_grpo/` | **沿用既有 `models/` `utils/` `scripts/` `configs/`** |
| Ablation 项数 | 4 | **7（cmd/anchor、N_anchor grid、噪声施加空间）+ Task 7.5 reference-KL** |
| 噪声施加空间 | 未指定（默认归一化） | **物理空间（denorm-perturb-renorm），避免 lateral/heading 近 0 时退化** |
| Inference 路径 | 假设可复用现有 FM 推理脚本 | **新写 `inference_action_navsim_anchored_grpo.py`，现有 FM 推理是 anchor-free** |
| 时间估算 | 4-5 周 | **5-6 周（Phase 6 重设计 + Phase 7 ablation 增多 + Task 0 真复现）** |

**核心结论**：v2.2 在 v2.1"工程上简化"的基础上做了**代码级一致性修订**，把 v2.1 中所有与实际仓库代码不符的假设（prefix、参数量、时间方向、注入方式、噪声形式、Mode Selector 结构）落到与 `models/policy_head/`、`reference/Emu3/`、`reference/DiffusionDriveV2/` 真实代码一致的形式。同时把 baseline 锁定从绝对 87.2 改为相对 B0 以应对 ckpt 真实状态不确定性（R11）。

---

## 9. 开发先后顺序（v2.2 落地建议）

按依赖链建议的执行顺序：

1. **Day 0**（必须先做）：Task 0.1-0.3。这一步决定 B0 与后续阈值。如果 B0 < 80，**先暂停后续开发**，跑完整 IL warm-up（直接复用 `train_pi0.py` + 完整 navtrain，目标到达 ≥ 85）后再回到 GRPO 主路径。
2. **Phase 1（解耦）**：Task 1.1-1.7 全是纯算法/工具模块，与 ckpt 解耦，单元测试驱动，可与 Phase 2 并行。
3. **Phase 2（核心改造）**：Task 2.1 → 2.3 → 2.2 → 2.4 → 2.5 → 2.6 顺序推荐——先把 anchored path（不动模型）做好，再注入 anchor token，最后做 sampler+log_pi。
4. **Phase 3（IL adapt）**：先 Task 3.2 的 λ_a warmup 必须真实跑通，再进 Task 3.4。
5. **Phase 4（GRPO main）**：在 Phase 3 ckpt 之上展开。
6. **Phase 5（risk）**：与 Phase 4 并行；R11/R12 在 Phase 0/3 阶段就要做，不要拖到 Phase 5 末尾。
7. **Phase 6（Mode Selector）**：Phase 4 的最优 ckpt 出来后才能开始（依赖 rollout 数据）。
8. **Phase 7（评估 + ablation）**：所有训练完成后。
