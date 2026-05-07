# RLActionToken Code Training Flow

本文按“训练入口 -> rollout 数据产生 -> replay buffer -> critic loss / actor loss”的顺序，说明当前仓库中 `RLActionToken` 的完整数据训练流向，并标注对应文件与行号。

> 行号基于本文编写时的当前工作树。若后续代码有增删，请以函数名为准重新定位。

---

## 0. 一句话总览

`RLActionToken` 的主训练路径是两阶段：

1. **Phase 1: encoder pretrain**
   冻结 VLA，从观察图像和语言指令中提取 VLA action-token hidden states，训练一个 encoder-decoder，把一段 VLA action queries 压缩成 `rl_token`，并用 reconstruction loss 保证信息不丢。

2. **Phase 2: off-policy TD3 RL**
   冻结 VLA 和 encoder，只训练小型 actor 和 twin-Q critic。rollout 时 VLA 提供 `action_queries` 和参考动作 `ã`，encoder 得到 `z_rl`，actor 产生实际动作 `a`，环境返回稀疏成功奖励。episode 被转换成 transition 后放进 replay buffer，训练线程从 buffer 采样，计算 critic TD loss 和 actor loss。

核心张量流向：

```text
image + instruction
  -> frozen VLA
  -> action_queries, vla_action ã
  -> encoder(action_queries) = rl_token z_rl
  -> actor(z_rl, prop_state, ã) = action a
  -> env.step_chunk(a) = next_obs, reward, done
  -> ReplayBuffer stores (z_rl, ã, a, r, z'_rl, ã', done, prop, prop')
  -> critic loss: MSE(Q(z_rl, a), target)
  -> actor loss: -Q(z_rl, actor(...)) + beta * ||a - ã||^2
```

---

## 1. 如何启动训练

推荐用现成脚本：

```bash
bash scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh 0,1,2,3,4,5
```

脚本会自动做两件事：

- 如果 `ENCODER_PATH` 不存在，先跑 `--phase pretrain`。
- 然后跑 `--phase rl_offpolicy`，也就是生产路径的 off-policy TD3 训练。

对应文件：

| 作用 | 文件和行号 |
|---|---|
| 设置 repo root、`.env`、`PYTHONPATH`、`LIBERO_PYTHON`、`LIBERO_HOME` | `scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh:3-13` |
| 设置 VLA SFT checkpoint、pretrain 输出目录、encoder path | `scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh:18-24` |
| Phase 1 pretrain 命令 | `scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh:31-63` |
| Phase 2 `rl_offpolicy` 命令 | `scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh:65-113` |

手动拆开跑也可以：

```bash
CUDA_VISIBLE_DEVICES=0 python AlphaBrain/training/reinforcement_learning/trainers/train.py \
  --phase pretrain \
  --ckpt_path results/training/QwenOFT-5traj-libero_goal/final_model \
  --output_dir results/rlt_training_TD3/5traj_alltasks_pretrain/pretrain \
  --suite libero_goal \
  --all_tasks \
  --bottleneck_dim 256 \
  --encoder_layers 2 \
  --encoder_heads 4 \
  --pretrain_n_obs 3000 \
  --pretrain_steps_per_reset 20 \
  --pretrain_epochs 500 \
  --pretrain_lr 1e-4 \
  --pretrain_batch_size 32 \
  --vla_extract_batch_size 16 \
  --num_envs_per_task 8 \
  --seed 42
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python AlphaBrain/training/reinforcement_learning/trainers/train.py \
  --phase rl_offpolicy \
  --ckpt_path results/training/QwenOFT-5traj-libero_goal/final_model \
  --encoder_path results/rlt_training_TD3/5traj_alltasks_pretrain/pretrain/checkpoints/pretrain_best/encoder.pt \
  --output_dir results/action_token_training_TD3/manual_run/rl_offpolicy \
  --suite libero_goal \
  --all_tasks \
  --use_steplock \
  --rollout_gpus 0,1,2,3,4 \
  --train_gpu 5 \
  --bottleneck_dim 256 \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 512 \
  --ref_dropout 0.5 \
  --fixed_std 0.1 \
  --G_per_task 30 \
  --num_envs_per_task 10 \
  --reward_coef 5.0 \
  --lr_actor 3e-4 \
  --lr_critic 3e-4 \
  --gamma 0.99 \
  --buffer_capacity 1000000 \
  --buffer_warmup 1024 \
  --warmup_iters 5 \
  --td_batch_size 1024 \
  --utd_ratio 10.0 \
  --tau 0.005 \
  --beta 1.0 \
  --actor_update_freq 2 \
  --target_noise_std 0.2 \
  --target_noise_clip 0.5 \
  --max_iter 400 \
  --seed 42
```

关键启动参数定义在 `AlphaBrain/training/reinforcement_learning/trainers/train_args.py`：

| 参数类别 | 文件和行号 |
|---|---|
| `--phase`, `--ckpt_path`, `--encoder_path`, task 选择 | `AlphaBrain/training/reinforcement_learning/trainers/train_args.py:7-21` |
| RLActionToken 结构参数 | `AlphaBrain/training/reinforcement_learning/trainers/train_args.py:23-34` |
| pretrain 参数 | `AlphaBrain/training/reinforcement_learning/trainers/train_args.py:36-47` |
| rollout / RL 基础参数 | `AlphaBrain/training/reinforcement_learning/trainers/train_args.py:48-81` |
| off-policy TD3 参数 | `AlphaBrain/training/reinforcement_learning/trainers/train_args.py:83-111` |
| GPU 划分参数 | `AlphaBrain/training/reinforcement_learning/trainers/train_args.py:122-128` |

---

## 2. 总入口如何分发 phase

入口文件是：

```text
AlphaBrain/training/reinforcement_learning/trainers/train.py
```

关键逻辑：

| 作用 | 文件和行号 |
|---|---|
| 文件注释说明三种 phase：`pretrain`、`rl`、`rl_offpolicy` | `AlphaBrain/training/reinforcement_learning/trainers/train.py:4-8` |
| import `parse_args`、`run_pretrain`、`run_rl_offpolicy`、`run_rl` | `AlphaBrain/training/reinforcement_learning/trainers/train.py:25-28` |
| 根据 `args.phase` 分发 | `AlphaBrain/training/reinforcement_learning/trainers/train.py:31-40` |

当前生产训练路径是：

```text
train.py
  -> parse_args()
  -> run_rl_offpolicy(args)
```

legacy on-policy 路径是 `--phase rl`，不是本文重点。

---

## 3. Phase 1: encoder pretrain 数据流

### 3.1 创建冻结 VLA 和 encoder-decoder

`run_pretrain(args)` 做这些事：

| 作用 | 文件和行号 |
|---|---|
| 加载 frozen VLA | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:26-31` |
| 读取 VLA hidden dim 和 chunk len | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:33-35` |
| 创建 `ActionTokenEncoderDecoder` | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:48-56` |
| 创建 optimizer | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:61` |

### 3.2 收集 observation，不训练 actor/critic

pretrain 阶段不跑 RL，也不训练 actor/critic。它只需要多样化 observation：

| 作用 | 文件和行号 |
|---|---|
| all tasks 时按 task 收集 observation | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:68-86` |
| single task 时收集 observation | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:87-99` |
| `collect_observations_fast()` reset env 并随机动作探索 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:253-348` |

收集到的数据形态：

```text
observations = [
  ([primary_image, wrist_image], instruction),
  ...
]
```

### 3.3 用 frozen VLA 提取 action queries

pretrain 会一次性把 observation 送进 frozen VLA，得到 action-token hidden states：

| 作用 | 文件和行号 |
|---|---|
| 调 `extract_action_queries_from_obs()` | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:101-108` |
| 批量调用 `frozen_vla.get_action_queries()` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:351-383` |
| QwenOFT 从最后 hidden states 中 gather action token embeddings | `AlphaBrain/model/framework/QwenOFT.py:218-257` |
| QwenOFT 的 gather 逻辑：取最后 `chunk_len` 个 action token | `AlphaBrain/model/framework/QwenOFT.py:277-328` |

张量形状：

```text
action_queries: (N, chunk_len, H)
```

其中：

- `N`: observation 数量。
- `chunk_len`: VLA 一次预测的动作长度。
- `H`: VLA hidden size。

### 3.4 encoder-decoder reconstruction loss

pretrain loss 在这里计算：

| 作用 | 文件和行号 |
|---|---|
| pretrain batch loop | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:118-161` |
| `_, recon_loss = enc_dec(batch_aq)` | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:133-135` |
| 保存最优 `encoder.pt` | `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:169-174` |

encoder-decoder 结构：

| 作用 | 文件和行号 |
|---|---|
| encoder 把 `(B, M, H)` 压缩成 `(B, 1, D)` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_encoder_decoder.py:33-84` |
| decoder 从 `rl_token` 重构 VLA tokens | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_encoder_decoder.py:87-164` |
| `encode()` 只调用 encoder | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_encoder_decoder.py:202-203` |
| forward 里计算 `recon_loss = mse(reconstructed, action_queries.detach())` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_encoder_decoder.py:208-224` |

公式上就是：

```text
action_queries z_{1:M}
  -> encoder
  -> z_rl
  -> decoder
  -> reconstructed z_{1:M}

L_recon = MSE(reconstructed, stop_gradient(action_queries))
```

Phase 1 结束后，RL 阶段只加载 encoder-decoder 的参数，但 rollout 和 TD3 主路径只用 `encoder.encode()`，不再用 decoder。

---

## 4. Phase 2: off-policy TD3 初始化

入口函数：

```text
run_rl_offpolicy(args)
```

对应 `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:42`。

### 4.1 GPU 划分和 frozen VLA

| 作用 | 文件和行号 |
|---|---|
| 解析 rollout GPUs 和 train GPU | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:51-59` |
| 每张 rollout GPU 加载 frozen VLA | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:65-74` |
| 如果需要，train GPU 上也加载 VLA | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:76-93` |
| 读取 `hidden_dim`、`chunk_len`、`action_dim`、action normalization stats | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:95-107` |

### 4.2 创建 encoder、actor、critic 和 target 网络

| 作用 | 文件和行号 |
|---|---|
| 创建 `ActionTokenEncoderDecoder` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:125-133` |
| 加载 `encoder_path` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:135-138` |
| 默认冻结 encoder，保证 buffer 里的 `rl_token` 表示稳定 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:140-149` |
| 创建 actor | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:150-158` |
| 创建 twin-Q critic | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:160-167` |
| 创建 target critic | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:169-173` |
| 创建 target actor | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:175-179` |
| 创建 rollout 用的轻量 `ActionTokenCritic`，只做 logging 兼容 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:181-186` |

actor 是 MLP Gaussian policy：

| 作用 | 文件和行号 |
|---|---|
| actor 输入维度：`bottleneck_dim + prop_dim + action_dim * chunk_len` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:49-50` |
| actor MLP | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:52-60` |
| 拼接 `(rl_token, prop_state, vla_action)` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:86-106` |
| 返回 Gaussian sample 和 log prob | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:125-135` |

critic 是 TD3 twin-Q MLP：

| 作用 | 文件和行号 |
|---|---|
| critic 输入维度：`bottleneck_dim + prop_dim + action_dim * chunk_len` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:179-180` |
| `q1` MLP 和 `q2` MLP | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:182-196` |
| critic 拼接 `(rl_token, prop_state, action)` 并输出 `Q1, Q2` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:198-218` |
| actor loss 只用 `q1_forward()` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:220-238` |

### 4.3 rollout 模块和 replay buffer

训练主进程维护一份 train GPU 上的 actor/critic，也会为 rollout GPU 拷贝 encoder/actor：

| 作用 | 文件和行号 |
|---|---|
| 为每张 rollout GPU deep copy encoder/actor/dummy critic | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:188-195` |
| `--use_steplock` 时创建 persistent env pools | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:209-245` |
| 非 steplock 时创建 `BatchInferenceServer` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:248-266` |
| 创建 actor 和 critic optimizer | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:280-286` |
| 创建中央 CPU replay buffer | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:303-304` |

---

## 5. rollout 时数据如何产生

默认脚本使用 `--use_steplock`，所以重点看：

```text
AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py
```

### 5.1 rollout 线程从训练循环启动

| 作用 | 文件和行号 |
|---|---|
| 定义后台 rollout 线程函数 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:520-662` |
| 根据 task 分配 rollout GPU | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:529-538` |
| steplock 多任务调用 `action_token_collect_multitask_steplock()` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:560-576` |
| steplock 单任务调用 `action_token_collect_group_steplock()` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:577-592` |
| 主线程启动 rollout 线程 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:670-675` |

### 5.2 单任务 steplock rollout

单任务函数是：

```text
action_token_collect_group_steplock()
```

关键数据流：

| 步骤 | 文件和行号 |
|---|---|
| reset 多个 env，得到初始 obs | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:112-129` |
| 创建 `ActionTokenEpisode` 容器 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:130-136` |
| 从 obs 中取 images、instruction、proprio state | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:154-159` |
| frozen VLA 输出 `action_queries, vla_actions` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:160-167` |
| encoder 得到 `rl_tokens` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:169-173` |
| actor chunk 比 VLA chunk 短时截断参考动作 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:175-179` |
| warmup 时直接用 VLA actions | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:181-185` |
| 非 warmup 时 actor 采样 actions | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:186-188` |
| 存 `ActionTokenStepRecord` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:197-210` |
| unnormalize action | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:214-220` |
| 并行执行 env chunk | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:222-247` |
| 根据环境 reward 设置 `success` 和 episode reward | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:239-246` |
| timeout/failure reward 置 0 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:272-279` |

在 steplock 路径中，环境执行一个 action chunk 的低层调用是：

| 作用 | 文件和行号 |
|---|---|
| `_env_step_chunk()` 把 chunk 内每步 action 后处理后送给 env pool | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:33-45` |
| env pool 的 `step_chunk()` 通过 worker 执行多个动作并返回 `obs, reward, done, steps_taken` | `AlphaBrain/training/reinforcement_learning/envs/persistent_env_pool.py:164-174` |

### 5.3 多任务 steplock rollout

多任务函数是：

```text
action_token_collect_multitask_steplock()
```

它和单任务版本相同，只是把多个 task 的 env 合并成一个 batch 做 VLA forward：

| 步骤 | 文件和行号 |
|---|---|
| 生成每个 task 的 episode/state 分配 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:335-348` |
| reset 全部 task/env | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:351-367` |
| 创建多任务 episodes | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:369-374` |
| merged batch VLA forward | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:385-393` |
| encoder 得到 `rl_tokens` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:395-396` |
| warmup 使用 VLA actions，非 warmup 使用 actor actions | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:398-409` |
| 存 `ActionTokenStepRecord` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:415-424` |
| env 并行执行 chunk，设置 episode reward | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:426-449` |
| finalize timeout/failure episodes | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:452-459` |

### 5.4 StepRecord 和 Episode 存了什么

数据结构在 `action_token_trainer.py`：

| 字段 | 含义 | 文件和行号 |
|---|---|---|
| `ActionTokenStepRecord.rl_token` | 当前 chunk 起点状态的 RL token | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:196-204` |
| `ActionTokenStepRecord.vla_action` | VLA 参考动作 chunk `ã` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:196-204` |
| `ActionTokenStepRecord.action_taken` | 实际执行的 actor/VLA 动作 chunk `a` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:196-204` |
| `ActionTokenStepRecord.prop_state` | 本体状态，默认 8 维 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:196-207` |
| `ActionTokenEpisode.reward` | episode 稀疏奖励 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:213-223` |
| `ActionTokenEpisode.success` | 是否成功 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:213-223` |
| `ActionTokenEpisode.finish_step` | episode 结束时已有多少个 step records | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:213-223` |
| `ActionTokenEpisode.done_cache_idx` | 终止发生在最后一个 chunk 内的位置 | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:213-223` |

---

## 6. episode 如何进入 ReplayBuffer

rollout 线程拿到 episodes 后立刻写入 replay buffer：

| 作用 | 文件和行号 |
|---|---|
| `push_episodes_to_buffer(all_eps, replay_buffer, gamma_per_step=args.gamma)` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:652-655` |
| 把 `(episodes, iteration, n_pushed)` 放进 queue 给主训练循环 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:656` |

### 6.1 transition 的字段

`ReplayBuffer.push()` 存的是：

```text
(
  rl_token,
  vla_action,
  action_taken,
  reward,
  next_rl_token,
  next_vla_action,
  done,
  prop_state,
  next_prop_state,
  task_id,
)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| ReplayBuffer 文档说明字段 | `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:1-10` |
| `push()` 参数列表 | `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:30-42` |
| transition 实际 tuple | `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:50-61` |
| `sample()` 返回前 9 个 tensor 字段 | `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:82-97` |
| `sample_balanced()` 多任务均衡采样 | `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:99-135` |
| `_collect()` stack 并搬到训练 device | `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:137-144` |

### 6.2 episode -> transition 的转换

转换函数：

```text
push_episodes_to_buffer()
```

关键逻辑：

| 作用 | 文件和行号 |
|---|---|
| 函数注释说明从 step records 转成 `(s, a, r, s', done)` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:801-824` |
| 计算 `stride_positions = [0, 2, 4, 6, ...]` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:832-835` |
| 根据 `done_cache_idx` 算最后 chunk 内的 `done_step` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:837-838` |
| 取当前 state: `rl_tok, vla_act, prop` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:851-862` |
| 构造长度为 C 的 action chunk | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:863-875` |
| 计算 sparse reward | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:876-881` |
| 构造 next state | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:883-898` |
| 调 `replay_buffer.push()` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:900-911` |

reward 逻辑：

```text
if terminal chunk and episode succeeded:
    r = gamma_per_step ** (done_step - p) * ep.reward
else:
    r = 0
```

其中：

- `p`: 当前 transition 从 chunk 内第几步开始。
- `done_step`: 成功/终止发生在最后 chunk 内第几步。
- `ep.reward`: 成功时为 `reward_coef`，失败时为 `0`。
- `gamma_per_step`: 单环境步折扣，通常是 `0.99`。

---

## 7. 训练循环如何消费 replay buffer

主训练循环在 `run_rl_offpolicy()` 里：

| 作用 | 文件和行号 |
|---|---|
| 从 rollout queue 阻塞取一批 episodes | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:686-696` |
| 统计成功率、reward、env steps | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:697-727` |
| warmup 阶段记录日志；实际是否 TD 更新继续由 buffer readiness 判断 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:728-747` |
| replay buffer ready 后进入 TD3 更新 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:744-759` |
| 每步先更新 critic | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:760-779` |
| 每 `actor_update_freq` 步更新 actor | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:780-795` |
| actor 更新后 soft-update target actor / critic | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:796-797` |
| 定期同步训练权重到 rollout 模块 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:812-816` |
| logging / wandb 记录 TD 指标 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:903-980` |

注意这里传入 critic update 的 gamma：

```python
gamma=args.gamma ** actor_chunk_len
```

对应 `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:762-774`。
也就是说，critic backup 的 bootstrap 项是 chunk 级折扣 `γ^C`。

---

## 8. critic loss 如何计算

函数：

```text
action_token_td_critic_update()
```

对应 `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1010-1076`。

### 8.1 从 replay buffer 采样 batch

```text
rl_tok, vla_act, act_taken, rew,
next_rl_tok, next_vla_act, done,
prop, next_prop = replay_buffer.sample(...)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| 多任务 balanced sample | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1038-1040` |
| 单任务/普通 sample | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1041-1043` |

### 8.2 target actor 生成 next action

```text
a' = μ_{θ'}(next_rl_tok, next_vla_act, next_prop)
```

代码里是：

```python
next_action, _ = target_actor(next_rl_tok, next_vla_act, next_prop, deterministic=True)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| 选择 target actor | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1045-1049` |
| TD3 target policy smoothing noise | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1050-1053` |

### 8.3 target critic 计算 TD target

代码：

```python
tq1, tq2 = target_q_critic(next_rl_tok, next_action, next_prop)
next_q = torch.min(tq1, tq2)
target = rew + gamma * next_q * (1.0 - done)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| target twin-Q | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1055-1057` |
| TD target | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1058` |
| target clipping | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1059-1064` |

公式：

```text
y_t = r_t + gamma_chunk * (1 - done_t) * min(Q1'(x_{t+C}, a'_{t+C}), Q2'(x_{t+C}, a'_{t+C}))
```

其中代码里的 `gamma_chunk` 已经是：

```text
gamma_chunk = args.gamma ** actor_chunk_len
```

### 8.4 online critic 拟合 target

代码：

```python
q1, q2 = q_critic(rl_tok, act_taken, prop)
critic_loss = mse(q1, target) + mse(q2, target)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| online Q forward | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1066-1068` |
| 返回 critic stats | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1070-1076` |
| 外层 `critic_loss.backward()` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:775` |
| critic optimizer step | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:776-778` |

---

## 9. actor loss 如何计算

函数：

```text
action_token_td_actor_update()
```

对应 `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1079-1128`。

### 9.1 从 replay buffer 采样 state 和 VLA reference

actor update 只需要：

```text
rl_tok, vla_act, prop
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| 多任务 balanced sample | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1104-1106` |
| 普通 sample | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1107-1109` |

### 9.2 当前 actor 采样动作

```python
action, _ = actor(rl_tok, vla_act, prop, deterministic=False)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| actor sampling | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1111-1112` |
| actor forward 中计算 Gaussian mean 和 sample | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:109-135` |

### 9.3 critic 给 actor 动作打分

```python
q_val = q_critic.q1_forward(rl_tok, action, prop)
```

对应 `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1114-1115`。

### 9.4 BC regularization 锚定到 VLA reference

```python
bc_penalty = ((action - vla_act) ** 2).sum(dim=(-2, -1)).mean()
actor_loss = -q_val.mean() + beta * bc_penalty
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| BC penalty | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1117-1118` |
| actor loss | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1120-1121` |
| 外层 `actor_loss.backward()` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:792` |
| actor optimizer step | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:793-795` |

公式：

```text
L_actor = -E[Q1(z_rl, prop, a_actor)] + beta * E[||a_actor - a_vla_ref||^2]
```

直觉：

- `-Q` 项让 actor 选择 critic 认为价值更高的动作。
- `beta * ||a - ã||^2` 项限制 actor 不要离 VLA 参考动作太远。

---

## 10. target actor / target critic 如何更新

target 网络在初始化时是当前网络的 deep copy：

| 作用 | 文件和行号 |
|---|---|
| 创建 target critic | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:169-173` |
| 创建 target actor | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:175-179` |

每次 actor 更新后，执行 Polyak averaging：

```python
soft_update_target(q_critic, target_q_critic, tau=args.tau)
soft_update_target(actor, target_actor, tau=args.tau)
```

对应文件和行号：

| 作用 | 文件和行号 |
|---|---|
| 外层调用 soft update | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:796-797` |
| soft update 实现：`target = (1 - tau) target + tau source` | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:151-155` |

---

## 11. warmup、rollout 权重同步、保存和评估

### 11.1 warmup

脚本设置了 `--warmup_iters 5`。warmup 期间 rollout 用 VLA 动作填 buffer，不用 actor 探索。代码注释里写了 warmup 会跳过 TD update，但当前控制流没有在 warmup 分支 `continue`；实际是否做 TD3 更新由 `replay_buffer.is_ready(min_size=args.buffer_warmup)` 决定。

| 作用 | 文件和行号 |
|---|---|
| 设置 warmup 模式 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:663-669` |
| steplock rollout 中 warmup 直接用 VLA actions | `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:181-185` |
| warmup 阶段记录日志 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:728-733` |
| buffer ready 后进入 TD3 更新 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:744-759` |
| warmup 结束后关闭 warmup 并同步 rollout weights | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:734-742` |

### 11.2 rollout 权重同步

训练主线程更新 actor 后，rollout GPU 上的 actor copy 需要同步：

| 作用 | 文件和行号 |
|---|---|
| `_sync_rollout_weights()` 定义 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:331-343` |
| 每隔 `sync_every_n_updates` 同步 | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:812-816` |

### 11.3 eval

评估使用 deterministic actor，也就是直接用 Gaussian mean，不采样噪声：

| 作用 | 文件和行号 |
|---|---|
| 同步 eval weights | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:356-365` |
| 多任务 eval 调 `_eval_deterministic_local()` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:376-447` |
| 单任务 eval 调 `_eval_deterministic_local()` | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:448-500` |
| 异步启动 eval thread | `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:862-881` |

### 11.4 checkpoint

保存的是 encoder、actor、critic：

| 作用 | 文件和行号 |
|---|---|
| checkpoint helper 保存 `encoder.pt`、`actor.pt`、`critic.pt` | `AlphaBrain/training/reinforcement_learning/common/ckpt_io.py:10-18` |

---

## 12. 按文件阅读顺序

建议按这个顺序看代码：

1. `scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh:31-113`
   先看训练命令到底传了什么参数。

2. `AlphaBrain/training/reinforcement_learning/trainers/train.py:31-40`
   看 phase 如何分发。

3. `AlphaBrain/training/reinforcement_learning/trainers/train_pretrain.py:22-178`
   看 encoder pretrain 如何收集 observation、提取 action queries、计算 reconstruction loss、保存 `encoder.pt`。

4. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_encoder_decoder.py:33-224`
   看 `action_queries -> rl_token -> reconstruction loss`。

5. `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:42-304`
   看 Phase 2 如何加载 frozen VLA、encoder、actor、critic、target 网络、replay buffer。

6. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_actor_critic.py:21-238`
   看 actor Gaussian policy 和 twin-Q critic MLP。

7. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:56-288`
   看单任务 steplock rollout 的数据如何从 VLA 进入 actor，再进入 env。

8. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_rollout_fast.py:291-465`
   看多任务 steplock rollout。

9. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:196-223`
   看 `StepRecord` 和 `Episode` 存了哪些字段。

10. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:801-913`
    看 episode 如何变成 replay transition。

11. `AlphaBrain/training/reinforcement_learning/common/replay_buffer.py:30-144`
    看 transition 如何存储和采样。

12. `AlphaBrain/training/reinforcement_learning/trainers/train_rl_offpolicy.py:686-817`
    看训练循环如何从 buffer 中触发 TD3 update。

13. `AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:1010-1128`
    看 critic loss 和 actor loss 的完整代码。

---

## 13. 最小 mental model

把当前实现记成下面三个对象就够了：

```text
VLA + encoder:
  obs -> action_queries -> rl_token

Actor:
  (rl_token, prop_state, vla_action_ref) -> action_chunk

Critic:
  (rl_token, prop_state, action_chunk) -> Q1, Q2
```

训练时：

```text
critic 学:
  buffer 里真实执行过的 action_chunk 到底能不能带来成功

actor 学:
  在 VLA 参考动作附近，怎样改 action_chunk 能让 critic 给更高 Q
```
