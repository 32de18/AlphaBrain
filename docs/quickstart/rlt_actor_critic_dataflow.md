# RL Token Actor / Critic 训练数据流

本文专门解释 AlphaBrain 当前 `RLActionToken` 实现中，actor/action policy 与 critic 的训练数据如何产生、如何进入 replay buffer，以及 TD3 更新时每个公式在代码中对应什么张量。

相关代码：

- `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py`
- `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py`
- `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py`
- `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py`

---

## 符号约定

RLActionToken 的控制单位是 **action chunk**，不是单步动作。

设：

```text
o_t        当前环境观测：图像、语言指令、proprio state
h_t        VLA action-query hidden states
z_t        encoder 输出的 RL token
s^p_t      proprioceptive state
x_t        RL critic/actor 使用的状态，x_t = (z_t, s^p_t)
ã_t        VLA reference action chunk
a_t        actor 输出并执行的 action chunk
r_t        chunk transition 的 reward
d_t        done 标记
C          actor action chunk 长度
A          action_dim
D          RL token bottleneck_dim
```

在默认 LIBERO / QwenOFT 配置中，常见形状是：

```text
h_t ∈ R^{C_vla × H}
z_t ∈ R^{1 × D}
ã_t ∈ R^{C × A}
a_t ∈ R^{C × A}

A = 7
D = 256
C_vla = 8
C = actor_chunk_len；不传 --actor_chunk_len 时 C = C_vla
```

---

## 1. Rollout 如何生成 actor / critic 训练数据

在线 RL 阶段的 rollout 路径由冻结 VLA、冻结 encoder 和当前 actor 组成。每当环境需要新的 action chunk，就执行：

```text
obs_t
  -> frozen VLA
       h_t, ã_t
  -> encoder
       z_t
  -> actor
       a_t
  -> unnormalize + postprocess
  -> LIBERO env.step / env.step_chunk
```

公式写成：

```text
h_t, ã_t = VLA(o_t)

z_t = g_φ(h_t)

x_t = concat(z_t, s^p_t)

a_t ~ π_θ(a | x_t, ã_t)
```

这里的 `ã_t` 是 VLA 预测的 normalized action chunk，也叫 reference action。它代表冻结 VLA 在当前观测下原本会执行的动作。RL actor 的职责不是从零学动作，而是在 VLA reference action 附近做局部改进。

代码中执行位置主要是：

- step-lock 快速路径：`action_token_rollout_fast.py` 中的 `action_token_collect_group_steplock()` / `action_token_collect_multitask_steplock()`。
- async server 路径：`action_token_trainer.py` 中的 `BatchInferenceServer` 和 `_action_token_rollout_one()`。

每个决策点会写入一个 `ActionTokenStepRecord`：

```text
rl_token      = z_t
vla_action    = ã_t
action_taken  = a_t
old_log_prob  = log π_θ(a_t | x_t, ã_t)
value         = legacy value logging
prop_state    = s^p_t
sub_tokens    = chunk 内 stride 位置的中间 token，视 rollout 路径而定
images        = 可选，仅 --finetune_vla 时保存
instruction   = 可选，仅 --finetune_vla 时保存
```

`ActionTokenEpisode` 再把多个 step records 加上 episode 级结果：

```text
reward
success
task_id
finish_step
env_steps
done_cache_idx
state_idx
video_path
```

---

## 2. Actor 的策略形式

代码类：`ActionTokenActor`

actor 输入：

```text
z_t       RL token
s^p_t     proprio state
ã_t       VLA reference action chunk
```

actor 输出完整动作 chunk：

```text
a_t ∈ R^{C × A}
```

策略分布是固定方差 Gaussian：

```text
π_θ(a_t | x_t, ã_t) = N( μ_θ(x_t, ã_t), σ²I )
```

其中：

```text
x_t = (z_t, s^p_t)
```

实现上，`ActionTokenActor._get_mean()` 会把三部分 flatten/concat：

```text
input_actor = concat(
    squeeze(z_t),       # D
    s^p_t,              # prop_dim，默认 8
    flatten(ã_t)        # C * A
)

μ_θ = MLP(input_actor).reshape(C, A)
```

训练采样时：

```text
a_t = μ_θ(x_t, ã_t) + σ ε,    ε ~ N(0, I)
```

推理 / deterministic eval 时：

```text
a_t = μ_θ(x_t, ã_t)
```

一个容易误解的点：当前代码里的 actor **不是结构 residual**。它不是：

```text
a_t = ã_t + Δa_t
```

而是：

```text
a_t = μ_θ(z_t, s^p_t, ã_t)
```

VLA reference action 只是 actor 的输入条件。actor 不偏离 VLA 的约束来自 actor loss 里的 BC regularization：

```text
β ||a_t - ã_t||²
```

训练时还会做 reference-action dropout：

```text
ã_t^{input} =
  0      with probability p_ref_dropout
  ã_t    otherwise
```

这防止 actor 只复制 `ã_t`，迫使它也利用 `z_t` 和 proprio state。

---

## 3. Episode 如何进入 ReplayBuffer

代码函数：`push_episodes_to_buffer()`

rollout 得到的是 episode records；TD3 训练需要 transition。`push_episodes_to_buffer()` 会把 episode 转成：

```text
(z_t, ã_t, a_t, r_t, z_{t+C}, ã_{t+C}, d_t, s^p_t, s^p_{t+C}, task_id)
```

对应 `ReplayBuffer.push()` 的字段：

```text
rl_token         z_t
vla_action       ã_t
action_taken     a_t
reward           r_t
next_rl_token    z_{t+C}
next_vla_action  ã_{t+C}
done             d_t
prop_state       s^p_t
next_prop_state  s^p_{t+C}
task_id          task index
```

critic 训练时使用：

```text
z_t, s^p_t, a_t, r_t, z_{t+C}, s^p_{t+C}, ã_{t+C}, d_t
```

actor 训练时使用：

```text
z_t, s^p_t, ã_t
```

默认 TD3 路径中，replay buffer 不保存原始图像，也不重新跑 VLA。图像只在 `--finetune_vla` 路径中保存，用来让梯度回传到 VLA。

### Chunk subsampling

函数注释里支持论文风格的 stride-2 chunk subsampling。对一个长度为 `C` 的 chunk，理论上可以从：

```text
p ∈ {0, 2, 4, 6}
```

这些位置构造 transition。位置 `p` 的 action 是跨 chunk 拼接：

```text
a_{t+p:t+p+C-1}
=
concat(
    current_chunk_action[p:C],
    next_chunk_action[0:p]
)
```

终止 chunk 上，如果成功 reward 出现在 chunk 内第 `done_step` 步，则位置 `p` 的 sparse reward 会折扣为：

```text
r_p = γ^{done_step - p} R_success
```

当前 async server 路径会在 chunk 内记录 `sub_tokens`，可以支持这些中间位置 transition。默认 step-lock 快速路径主要记录 chunk 起点，所以实际更接近 chunk-start transition。

---

## 4. Critic 训练数据流

代码函数：`action_token_td_critic_update()`

critic 是 TD3 的 twin-Q：

```text
Q_{ψ1}(x_t, a_t), Q_{ψ2}(x_t, a_t)
```

其中：

```text
x_t = (z_t, s^p_t)
```

输入拼接方式：

```text
input_Q = concat(
    squeeze(z_t),       # D
    s^p_t,              # prop_dim
    flatten(a_t)        # C * A
)
```

训练 batch 从 replay buffer 采样：

```text
B = {
  z_t, ã_t, a_t, r_t,
  z_{t+C}, ã_{t+C}, d_t,
  s^p_t, s^p_{t+C}
}
```

### 4.1 Target action

先用 target actor 在 next state 上生成 next action：

```text
a'_{t+C} = μ_{θ'}(x_{t+C}, ã_{t+C})
```

TD3 还会加 target policy smoothing noise：

```text
ε ~ clip(N(0, σ_target²), -c, c)

ā'_{t+C} = clip(a'_{t+C} + ε, -1, 1)
```

代码对应：

```python
next_action, _ = target_actor(
    next_rl_tok,
    next_vla_act,
    next_prop,
    deterministic=True,
)

noise = torch.randn_like(next_action) * target_noise_std
noise = noise.clamp(-target_noise_clip, target_noise_clip)
next_action = (next_action + noise).clamp(-1.0, 1.0)
```

### 4.2 TD target

target critic 计算 twin-Q，然后取较小值：

```text
Q'_{min}
=
min(
  Q_{ψ1'}(x_{t+C}, ā'_{t+C}),
  Q_{ψ2'}(x_{t+C}, ā'_{t+C})
)
```

chunk-level TD target：

```text
y_t = r_t + γ^C (1 - d_t) Q'_{min}
```

代码里 `train_rl_offpolicy.py` 调 critic update 时传的是：

```python
gamma=args.gamma ** actor_chunk_len
```

所以 `action_token_td_critic_update()` 内部的：

```python
target = rew + gamma * next_q * (1.0 - done)
```

实际就是：

```text
target = r_t + γ^C (1 - d_t) Q'_{min}
```

### 4.3 Critic loss

当前 critic 对 buffer 里的真实执行动作 `a_t` 做回归：

```text
L_Q(ψ)
=
E_B [
  (Q_{ψ1}(x_t, a_t) - y_t)^2
  +
  (Q_{ψ2}(x_t, a_t) - y_t)^2
]
```

代码对应：

```python
q1, q2 = q_critic(rl_tok, act_taken, prop)

critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
```

critic 学到的是：**在 RL token 状态和 proprio state 下，执行某个 action chunk 的长期价值**。

---

## 5. Actor 训练数据流

代码函数：`action_token_td_actor_update()`

actor update 也从 replay buffer 采样，但它不使用 buffer 里的旧 `action_taken` 来做 supervised regression。它使用 buffer 的状态和 VLA reference action，重新生成当前策略动作：

```text
a_θ ~ π_θ(a | x_t, ã_t)
```

然后用当前 critic 的 Q1 评分：

```text
Q_{ψ1}(x_t, a_θ)
```

actor loss：

```text
L_π(θ)
=
E_B [
  - Q_{ψ1}(x_t, a_θ)
  +
  β ||a_θ - ã_t||²
]
```

第一项：

```text
-Q_{ψ1}(x_t, a_θ)
```

表示最大化 critic 估计的 return。

第二项：

```text
β ||a_θ - ã_t||²
```

表示把 actor 锚定在 VLA reference action 附近。`β` 越大，actor 越保守；`β` 越小，actor 越容易偏离 VLA。

代码对应：

```python
action, _ = actor(
    rl_tok,
    vla_act,
    prop,
    deterministic=False,
)

q_val = q_critic.q1_forward(rl_tok, action, prop)

bc_penalty = ((action - vla_act) ** 2).sum(dim=(-2, -1)).mean()

actor_loss = -q_val.mean() + beta * bc_penalty
```

注意这里的 BC 正则不是行为克隆 dataset 上的 supervised loss，而是相对当前 VLA reference action 的 anchor。它的作用是限制在线 RL 的搜索范围，让 policy refinement 发生在 VLA 已有行为附近。

---

## 6. TD3 更新节奏

主循环在 `train_rl_offpolicy.py` 中。

每轮 iteration 的结构：

```text
rollout thread:
  current actor -> collect episodes -> push replay buffer

train thread:
  sample replay buffer
  critic update
  delayed actor update
  soft update target networks
  periodically sync actor/encoder weights to rollout GPUs
```

### 6.1 Warmup

前 `warmup_iters` 轮使用 pure VLA action 采样：

```text
a_t = ã_t
```

这样 replay buffer 先被 VLA 的有效行为填充，避免随机初始化 actor 直接控制环境导致全是低质量数据。

### 6.2 Update-to-data ratio

每轮 rollout 产生 `n_new_transitions` 后，TD update 次数大致是：

```text
n_updates
=
floor(n_new_transitions * utd_ratio / batch_size)
```

并受 `td_updates_per_iter` 上限限制：

```text
n_updates = min(n_updates, td_updates_per_iter)
```

### 6.3 Delayed actor update

critic 每个 TD step 都更新；actor 每隔 `actor_update_freq` 个 critic step 更新一次：

```text
if (td_step + 1) mod actor_update_freq == 0:
    update actor
    soft update target actor / target critic
```

target network soft update：

```text
ψ' ← (1 - τ) ψ' + τ ψ

θ' ← (1 - τ) θ' + τ θ
```

代码：

```python
soft_update_target(q_critic, target_q_critic, tau=args.tau)
soft_update_target(actor, target_actor, tau=args.tau)
```

---

## 7. 一条完整数据链

完整训练数据流可以压缩成：

```text
LIBERO obs
  -> frozen VLA
       h_t, ã_t
  -> encoder
       z_t
  -> actor
       a_t
  -> env executes a_t
       reward, next obs, done
  -> replay buffer
       (z_t, ã_t, a_t, r_t, z_{t+C}, ã_{t+C}, done)

critic update:
  sample buffer
  -> target actor produces next action
  -> target critic builds TD target
  -> twin-Q MSE update

actor update:
  sample buffer
  -> actor produces new action
  -> critic scores new action
  -> maximize Q while staying close to VLA reference
```

用一句话概括：

```text
critic 学的是 “在 RL token 状态下，一个 action chunk 的长期价值”；
actor 学的是 “在 VLA 推荐动作附近，怎样微调 action chunk 让 critic 估计的回报更高”。
```

---

## 8. 与代码字段的对应表

| 数学符号 | 代码字段 / 变量 | 说明 |
|:--|:--|:--|
| `o_t` | `obs` | LIBERO observation，包含 `primary_image`、`wrist_image`、`state`。 |
| `h_t` | `action_queries` | VLA action-token hidden states。 |
| `z_t` | `rl_token` / `rl_tok` | encoder 输出的 RL token。 |
| `s^p_t` | `prop_state` / `prop` | proprioceptive state，默认 8 维。 |
| `x_t` | `(rl_tok, prop)` | actor/critic 实际使用的 state。 |
| `ã_t` | `vla_action` / `vla_act` | VLA reference action chunk。 |
| `a_t` | `action_taken` / `act_taken` | rollout 中实际执行的 action chunk。 |
| `a_θ` | `action` | actor update 中当前 actor 重新生成的 action。 |
| `r_t` | `reward` / `rew` | transition reward。 |
| `d_t` | `done` | terminal mask。 |
| `z_{t+C}` | `next_rl_token` / `next_rl_tok` | next RL token。 |
| `ã_{t+C}` | `next_vla_action` / `next_vla_act` | next VLA reference action。 |
| `Q_{ψ1}, Q_{ψ2}` | `q_critic.q1`, `q_critic.q2` | online twin-Q。 |
| `Q_{ψ1'}, Q_{ψ2'}` | `target_q_critic` | target twin-Q。 |
| `π_θ` | `actor` | online actor。 |
| `π_{θ'}` | `target_actor` | target actor。 |
| `β` | `args.beta` | VLA reference BC 正则权重。 |
| `τ` | `args.tau` | target network soft update 系数。 |
