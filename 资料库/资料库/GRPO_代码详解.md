# GRPO_ippo_rnn_JAXMARL_pmap.py 代码详解

## 文件概述

`GRPO_ippo_rnn_JAXMARL_pmap.py` 是基于 **GRPO（Group Relative Policy Optimization）** 的多智能体训练入口。它实现了 **Actor-only IPPO**（去掉 critic，用 group advantage 替代 GAE），支持多智能体类型（MM + EXE），使用 `ScannedLSTM` 处理时序，通过 `pmap` 在 4 张 GPU 上并行。

---

## 一、总体框架

```
make_train(config)
    │
    ├── 环境初始化 (MARLEnv)
    ├── 网络初始化 (ActorOnlyRNN × agent类型数)
    ├── 训练循环 train(rng)
    │       │
    │       ├── pmap 设备分发
    │       ├── jax.lax.scan 主循环 (_update_step × NUM_UPDATES)
    │       │       │
    │       │       ├── _env_step (收集 NUM_STEPS 步轨迹)
    │       │       │       ├── 策略采样 → 动作
    │       │       │       ├── env.step 执行
    │       │       │       └── 存 GRPOTransition
    │       │       │
    │       │       ├── group advantage 计算
    │       │       │       └── (episode_return - mean) / std
    │       │       │
    │       │       └── _update_epoch (× UPDATE_EPOCHS)
    │       │               └── _update_minbatch (× NUM_MINIBATCHES)
    │       │                       └── _loss_fn → grad → apply_gradients
    │       │
    │       └── callback → wandb.log + print
    │
    └── 返回训练结果
```

---

## 二、关键组件

### 1. ScannedLSTM

```python
class ScannedLSTM(nn.Module):
```

- 用 `nn.scan` 在时间步上展开 LSTM，参数在所有时间步共享
- 在 episode 边界（`dones=True`）重置 hidden/cell 为 0
- 输入: `(hidden, cell), (embedding, dones)`
- 输出: `(new_hidden, new_cell), y`

标准 LSTM 每个时间步输出一个 hidden state，`ScannedLSTM` 通过 `nn.scan` 一次性处理整个序列（NUM_STEPS 步），JAX 会自动编译成高效循环。

### 2. ActorOnlyRNN

```python
class ActorOnlyRNN(nn.Module):
```

- **无 critic**，只有 actor 网络
- 架构: `obs → Dense(128, relu) → ScannedLSTM → Dense(128, relu) → Dense(action_dim) → Categorical`
- 输出 `distrax.Categorical` 分布（离散动作的概率分布）
- 每种 agent 类型（MM、EXE）有独立的 ActorOnlyRNN 实例

### 3. GRPOTransition

```python
class GRPOTransition(NamedTuple):
    global_done, done, action, reward, log_prob, obs, info
```

- 存储单步轨迹数据
- 每个 agent 类型独立存储
- 维度: `(NUM_STEPS, NUM_ACTORS, ...)`，其中 `NUM_ACTORS = NUM_ENVS × NUM_AGENTS_PER_TYPE`

### 4. batchify / unbatchify

```python
def batchify(x, num_actors):    # (NUM_ENVS, NUM_AGENTS, ...) → (NUM_ACTORS, ...)
def unbatchify(x, num_envs, num_agents):  # 反向
```

- 在环境维度和 agent 维度之间切换
- 环境交互时用 `unbatchify` 拆成 `(NUM_ENVS, NUM_AGENTS)` 传入 env.step
- 策略计算时用 `batchify` 合并成 `(NUM_ACTORS,)` 扁平处理

---

## 三、GRPO 损失计算（核心）

在 `_update_minbatch` 的 `_loss_fn` 中：

```python
def _loss_fn(params, init_hstate, traj_batch, advantages, ref_params):
    # 1. 用当前策略重新计算 log_prob
    _, pi = train_state.apply_fn(params, ...)
    log_prob = pi.log_prob(traj_batch.action)

    # 2. 用参考策略计算 log_prob（用于 KL）
    _, pi_ref = train_state.apply_fn(ref_params, ...)
    log_prob_ref = pi_ref.log_prob(traj_batch.action)

    # 3. Actor loss：PPO clip 损失
    logratio = log_prob - traj_batch.log_prob   # 新旧策略比值
    ratio = exp(logratio)
    loss_actor1 = ratio * advantages
    loss_actor2 = clip(ratio, 1-ε, 1+ε) * advantages
    loss_actor = -min(loss_actor1, loss_actor2).mean()

    # 4. Entropy bonus（鼓励探索）
    entropy = pi.entropy().mean()

    # 5. KL penalty（防止偏离参考策略）
    lograt = log_prob_ref - log_prob
    kl_penalty = mean(exp(lograt) - lograt - 1.0)

    # 6. 总损失
    total_loss = loss_actor + KL_COEF × kl_penalty - ENT_COEF × entropy
```

### 损失函数详解

#### Actor Loss (PPO clip)

$$
L_{\text{actor}} = -\mathbb{E}\left[\min(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)\right]
$$

- $r_t(\theta) = \exp(\log \pi_{\text{new}}(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t))$ 是新旧策略的概率比
- $\epsilon = 0.2$ 是 clip 边界
- $r_t < 1-\epsilon$ 或 $r_t > 1+\epsilon$ 时梯度被裁剪，防止单步更新过大

#### KL Penalty（GRPO 特色）

$$
KL_{\text{penalty}} = \mathbb{E}\left[\frac{\pi_{\text{ref}}(a|s)}{\pi_{\text{current}}(a|s)} - \log\frac{\pi_{\text{ref}}(a|s)}{\pi_{\text{current}}(a|s)} - 1\right]
$$

- 标准 PPO 用 clip 限制更新幅度，GRPO 额外加 KL 惩罚
- 参考策略是**本次 update step 开始时的 snapshot**，在 4 个 epoch 内固定
- `ref_params_list[i]` 在 `_update_step` 开始时 snapshot，在 `_update_epoch` 外捕获

#### Entropy Bonus

$$
L_{\text{entropy}} = -\beta \cdot \mathbb{E}[H(\pi(\cdot|s))]
$$

- $\beta = \text{ENT\_COEF}$
- 负号让 loss 对 entropy 负相关 → 最小化 loss 鼓励高 entropy（探索）

#### GRPO Advantage（无 critic）

$$
\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}
$$

- $R_i$ 是 agent i 的整局总回报（`episode_return = sum(reward)`）
- $\mu_R, \sigma_R$ 是所有该 type agent 回报的均值和标准差
- **不使用时序差分（GAE）**，不使用价值网络

---

## 四、ENT_COEF 的作用

### 数学位置

```python
total_loss = loss_actor + KL_COEF × kl_penalty - ENT_COEF × entropy
```

熵（entropy）衡量策略的随机性：

| entropy 值 | 策略行为 |
|:-----------|:---------|
| 高 (~1.45) | 接近均匀随机，各种动作概率差不多 |
| 中 (~0.5) | 有一定偏向，但仍保留探索 |
| 低 (~0.01) | 几乎确定性的，只选 1-2 个动作 |

### 当 ENT_COEF 过大时 — 你遇到的问题

```
entropy_loss = ENT_COEF × entropy
             = 0.001 × 1.45
             = 0.00145

actor_loss ≈ -8.6e-06

total_loss ≈ 0.00145 + (-8.6e-06) ≈ 0.00145
           ≈ entropy_loss
```

- actor_loss 只占 0.6%，**entropy_loss 占 99.4%**
- 梯度方向由 entropy 主导 → 策略被推向均匀分布
- 新旧策略差异极小 → ratio ≈ 1.0 → clip_frac ≈ 0 → **策略锁死**

### 当 ENT_COEF=0 时

```
total_loss = loss_actor + KL_COEF × kl_penalty
```

- 梯度完全由 reward 驱动
- actor_loss 即使只有 1e-5，也能推动策略更新
- 风险：策略可能过早确定化（collapse），停止探索

### ENT_COEF 的权衡

```
ENT_COEF 太高                 ENT_COEF=0
    │                            │
    ▼                            ▼
探索过多，不利用         探索过少，过早收敛
更新≈0，奖励不涨         可能过拟合到噪声
    │                            │
    └──────── 合适值 ────────────┘
            ~1e-5 到 1e-4
```

---

## 五、Group Advantage 的抵消效应

### 推导

`reward_scaling_quo` 改变 reward 量级，但被 group advantage 归一化抵消：

```
假设所有人 reward 乘以 c（因为 quo 变小）:
    R_i' = c × R_i
    μ' = c × μ
    σ' = c × σ

    A_i' = (cR_i - cμ) / (cσ + ε) ≈ (R_i - μ) / σ = A_i
```

这解释了为什么两次实验（quo=100 vs quo=20）的 `actor_loss` 几乎一样。

### 打破锁死的路径

1. **降低 ENT_COEF** → 减少 entropy 在 loss 中的占比
2. **降低 KL_COEF** → 放松 KL 约束，允许策略更新
3. **换 PPO with critic** → GAE advantage 不受 group normalize 抵消
4. **去掉 group normalize** → 直接用 raw reward 做 advantage

---

## 六、数据流维度变化

```
环境交互阶段:
  obs / action / reward: (NUM_ENVS, NUM_AGENTS, feat_dim)
  batchify → (NUM_ACTORS, feat_dim)     [NUM_ACTORS = NUM_ENVS × NUM_AGENTS]

网络计算阶段:
  (NUM_STEPS, NUM_ACTORS, feat_dim) → ScannedLSTM → Categorical

pmap 分发:
  每设备拿 NUM_ACTORS/N_DEVICES 份，即 (NUM_ENVS/N_DEVICES × NUM_AGENTS)

advantage 计算:
  traj_batch.reward: (NUM_STEPS, NUM_ACTORS/N_DEVICES)
  episode_return: (NUM_ACTORS/N_DEVICES,) — sum over steps
  advantage: (1, NUM_ACTORS/N_DEVICES) — 广播到每一步
```

---

## 七、key 配置参数

| 参数 | 值 | 作用 |
|:-----|:---|:-----|
| NUM_ENVS | 16384 | 并行环境数 |
| NUM_STEPS | 64 | 每条轨迹长度 |
| UPDATE_EPOCHS | 4 | 每次收集后重放次数 |
| NUM_MINIBATCHES | 32 | minibatch 切分数 |
| TOTAL_TIMESTEPS | 2e8 | 总训练步数 |
| ENT_COEF | [0.001, 0.001] | entropy 权重 |
| KL_COEF | [0.01, 0.01] | KL 惩罚权重 |
| CLIP_EPS | 0.2 | PPO clip 边界 |
| LR | [1e-4, 1e-4] | 学习率，annealed |
| N_DEVICES | 4 | GPU 数量 |
| GRU_HIDDEN_DIM | 128 | LSTM hidden size |
| FC_DIM_SIZE | 128 | 特征提取层大小 |
