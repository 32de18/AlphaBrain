# RLActionToken — 设计笔记 & 与 RL Token 论文的差异

本目录包含我们对 VLA 模型的 off-policy、bottleneck-token 在线 RL 方案的实现，名称为 **`RLActionToken`**。该名称特意与 **RL Token** 论文（Physical Intelligence, 2026）区分开来——论文的高层思路启发了本模块，但我们**并非**逐行复现论文的实现。

本 README 说明本目录的内容、各部分的协作关系，以及——最重要的——**我们与论文的差异**，以便读者将本代码与论文对照时不至于困惑。

---

## 文件结构

| 文件 | 功能 |
|:-----|:-----|
| `action_token_encoder_decoder.py` | `ActionTokenEncoder` 和 `ActionTokenDecoder`：瓶颈编码器 + 重建解码器，用于 Phase 1 预训练 |
| `action_token_actor_critic.py` | TD3 actor `μ_θ(x, ã)` 和双 critic `Q_{ψ1,2}(x, a)` |
| `action_token_trainer.py` | 损失项、TD 更新步骤、检查点管理 |
| `action_token_rollout_fast.py` | 多环境并行步锁批量 rollout；为 replay buffer 喂数据 |
| `__init__.py` | 重新导出公共接口 |

训练器入口在上一级 `trainers/` 目录中；驱动端到端运行的脚本在 `scripts/run_rl_scripts/` 中。

---

## 高层架构

```
obs ──► 冻结的 VLA ──► action-query 隐状态 (M × H)
                              │
                              ▼
                         ActionTokenEncoder  ──►  z_rl ∈ ℝ^{1 × D}
                              │                        │
           Phase 1: ─────────►│                        │
           ActionTokenDecoder ◄───  重建 sg(action_queries)
                                                       │
           Phase 2: ─────────────────────────────────► │
                                       x = (z_rl, s_p)
                                              │
                                              ▼
                                    TD3 actor / 双 critic
                                       (off-policy 训练)
```

- **Phase 1** 用重建目标训练编码器/解码器。
- **Phase 2** 冻结 VLA（及可选的编码器），在 rollout worker 收集的转移数据上训练小型 actor/critic。

---

## 与 RL Token 论文的差异

`RLActionToken` 保留了论文的高层架构——冻结的 VLA 向小型 TD3 actor/critic 输入紧凑状态——但在多个具体选择上有所不同。如果你在将本代码与论文对照，请先阅读本节，不要默认两者等价。

### 编码器 (`ActionTokenEncoder`)

- **输入。** 我们使用 VLA 的 **action-query 隐状态**
  `action_queries ∈ ℝ^{B × M × H}`（从 `last_hidden` 的 action-token 位置提取），其中 `M = chunk_len = 8`，`H = 2048`（Qwen2.5-VL-3B）。而论文 Fig. 2 使用的是 **完整 image-token 嵌入** `N × 2048`，来自 VLM backbone。
- **结构。** 我们在序列末尾追加一个可学习的 `e_rl` CLS token，通过一个小型自注意力编码器（默认 2 层、4 头、`d_model = H`），然后取 `e_rl` 位置的输出。
- **额外瓶颈投影。** 编码器之后施加 `Linear(H → bottleneck_dim)`，得到 `z_rl ∈ ℝ^{B × 1 × D}`，默认 `D = 256`。**论文保持 `z_rl` 在 VLA 隐维度（`1 × 2048`）**——其瓶颈来自将 `N` 个 token 压缩为 `1` 个 token，而非降低每个 token 的宽度。我们的额外投影是为下游小型 actor/critic MLP 提供的一个实用调节旋钮，但这是一个实质性的偏差。

### 解码器 (`ActionTokenDecoder`)

- 仅在 Phase-1 预训练期间使用，用于重建损失以强制瓶颈保持信息量。
- 我们用 `Linear(D → H)` 将 `z_rl` 扩展回 VLA 隐维度，将其作为前缀拼接到 teacher-forced 移位 VLA token 序列前，加上可学习位置嵌入，然后运行因果掩码自注意力栈（`TransformerEncoderLayer`，`src_mask = triu`，默认 2 层）。
- **这是前缀 + 因果自注意力方案，而非论文 Eq. 2 展示的编码器-解码器交叉注意力结构。** 功能相近，架构不同。
- `L_ro = MSE(reconstructed, sg(action_queries))` — 对所有重建位置的 MSE，VLA token 侧加 stop-gradient。

### 其他值得注意的差异

- **预训练数据。** 论文使用任务演示 `D`。我们目前通过在环境中用**随机动作** rollout 收集观测（`collect_observations_fast`），以求简便；这与论文的数据分布不匹配。
- **预训练时联合微调 VLA。** 论文 Algorithm 1（第 3 行）优化 `ϕ, θ_vla = argmin L_ro(ϕ) + α L_vla(θ_vla)`。我们的 Phase-1 仅训练编码器/解码器；`--finetune_vla` 存在但在 RL 阶段才激活，不在编码器预训练期间。
- **基础 VLA。** 论文使用 `π0.6`（SigLIP + Gemma 4B + 860M flow/diffusion action expert）。我们基于 QwenOFT（Qwen2.5-VL-3B + MLP action head），因此 VLA 的参考动作是单峰的，而非多峰扩散采样——`ref-action pass-through` 仍然有帮助，但原因与论文不同。
- **环境 / 分块。** 论文：14 维双臂真实世界，RL 分块 `C = 10`，VLA `H = 50`，50 Hz。我们发布：7 维 LIBERO 仿真，VLA 分块 `= 8`，actor 分块 `= 4`。
- **人在回路、关键阶段切换、策略交接学习** — 均仅在论文中出现。当前发布版本是纯自主仿真方案。

### 与论文一致的部分

- TD3 off-policy，双 Q 值和目标策略平滑。
- Actor `μ_θ(x, ã)` + 固定小高斯标准差，BC 正则项
  `β ‖a − ã‖²`，50% 参考动作 dropout。
- 推入 replay buffer 时以步长 2 对分块子采样。
- `x = (z_rl, s_p)` 状态构建，其中 `s_p` 为本体感知。

---

## "官方"论文精确实现

我们正在积极测试一条更严格遵循论文的路线——image-token 输入、`1 × 2048` RL token 无额外投影、交叉注意力解码器、演示驱动的 Phase-1、联合 VLA 微调。目前**尚不够稳定，无法发布**；届时将作为同级模块发布而非替代，以避免破坏现有 `RLActionToken` 用户。

---

## 入口

- **训练（Phase 1 + Phase 2）**：`scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh`
- **评估**：`scripts/run_rl_scripts/run_eval_action_token.sh`
- **配方 YAML**：`configs/rl_recipes/QwenOFT_LIBERO_ActionToken.yaml`
- **脚本级 README**（CLI 参数、rollout 数学、注意事项）：`scripts/run_rl_scripts/README.md`
