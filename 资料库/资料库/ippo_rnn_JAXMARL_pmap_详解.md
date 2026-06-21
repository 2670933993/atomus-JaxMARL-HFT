# IPPO + RNN + PMAP 训练脚本详解

> 文件：`gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL_pmap.py`
> 论文：JaxMARL-HFT (Oxford 2026) — 基于 PureJaxRL 的 IPPO 实现
> 注意：本文件是基于我对克隆仓库的理解编写的注解，并非原始论文官方文档。

---

## 一、总览：这个文件在做什么

这是整个项目的**核心训练脚本**。它实现了：

```
多个 Agent 类型（MM 做市商 + EXE 执行端）
        ↓ 在共享订单簿环境中交互
   IPPO + RNN 算法训练策略网络
        ↓ 通过 PMAP 分布在多张 GPU 上
   并行收集 trajectory → 计算 GAE → 更新网络
```

**输入：** YAML 配置文件（hyperparameters + agent configs + data paths）
**输出：** 训练好的策略网络参数 + wandb 日志

---

## 二、文件内依赖关系（自上而下）

```
main(config)                         ← 入口 1 (Hydra + wandb sweep)
  │
  ├── make_train(config)              ← 核心工厂函数，返回 train(rng)
  │     │
  │     ├── ScannedRNN                ← GRU 时序扫描模块
  │     ├── ActorCriticRNN            ← Actor-Critic 神经网络
  │     ├── Transition                ← 轨迹数据结构
  │     ├── batchify / unbatchify     ← 形状变换工具
  │     │
  │     └── train(rng)                ← 实际训练循环
  │           │
  │           ├── 初始化网络 / 优化器 / 环境
  │           ├── _env_step()         ← 单步环境交互
  │           ├── _calculate_gae()    ← GAE advantage 计算
  │           ├── _loss_fn()          ← IPPO 损失函数
  │           ├── _update_minbatch()  ← 单 minibatch 更新
  │           ├── _update_epoch()     ← 多 epoch 更新
  │           ├── _update_step()      ← 完整更新步
  │           └── callback()          ← wandb 日志
  │
  └── sweep_fun()                     ← wandb sweep 包装
```

---

## 三、核心算法：IPPO（Independent PPO）

### 3.1 什么是 IPPO

传统 PPO 训练单一 agent，IPPO 训练**多个独立**的 agent，**各自有独立的策略网络和价值网络**，但共享同一环境。

| 维度 | 普通 PPO | IPPO |
|:----|:--------|:-----|
| agent 数量 | 1 | N |
| 策略网络 | 1 | N (每类 agent 各 1 个) |
| 价值网络 | 1 | N |
| 环境共享 | — | 是（通过同一 LOB） |
| 梯度通信 | 无 | 无（各自独立更新） |

核心公式（对每个 agent 独立计算）：

```math
L^CLIP(θ) = E_t[ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]
                ↑ PPO 裁剪后的代理目标函数
```

### 3.2 这里的 IPPO 具体实现

你配置了 `NUM_AGENTS_PER_TYPE: [5, 5]`，表示：

```
MM 类: 5 个独立 agent (共享同一组网络参数)
EXE 类: 5 个独立 agent (共享另一组网络参数)
        ↓
总共 10 个 agent × 16384 环境 = 163840 个并行 actor
```

重点：**同一类的 agent 共享网络参数**（权重相同），但各自有独立的 hidden state（RNN 记忆不同）。这叫 parameter-sharing IPPO。

---

## 四、神经网络架构

### 4.1 ScannedRNN — GRU 时序模块

```python
class ScannedRNN(nn.Module):
    @nn.scan                          # ← 沿时间轴展开
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        # 如果 episode 结束 (done)，重置 hidden state 到 0
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y
```

**关键点：**
- `nn.scan` — JAX 的循环展开原语，沿时间步展开 GRU 计算（不写 for 循环，一次编译全部展开）
- `carry` — 时间上的 RNN hidden state
- `resets` — 当 episode 结束时重置 hidden state（让不同 episode 的记忆不混杂）
- `nn.GRUCell` — 单步 GRU 模块

### 4.2 ActorCriticRNN — 完整网络

```
输入: hidden_state, (obs, dones)
       │
       ▼
 Dense(128) + ReLU ← FC_DIM_SIZE[默认128]
       │
 ScannedRNN(GRU, hidden=128) ← GRU_HIDDEN_DIM[默认128]
       │
  ┌────┴────┐
  │         │
 Actor头    Critic头
  │         │
 Dense(128)  Dense(128)
  +ReLU       +ReLU
  │         │
 Dense(n_actions) Dense(1)
  │         │
 Categorical    value
 (pi)
```

**共享 GRU 的设计含义：**
Actor 和 Critic **共享底层特征提取**（Dense + GRU），这意味着：
- 梯度更新时，Critic 的更新会影响 Actor 使用的特征向量
- 这是一种**参数效率高但可能有梯度冲突**的设计
- 近期研究倾向于分离 Actor/Critic 的编码器

**输出三样东西：**
1. `hidden` — 更新后的 GRU hidden state（传给下一时间步）
2. `pi` — 动作概率分布（`distrax.Categorical`）
3. `value` — 状态价值估计（标量，用于 GAE 计算和价值损失）

---

## 五、核心数据结构

### 5.1 Transition（轨迹步）

```python
class Transition(NamedTuple):
    global_done: jnp.ndarray   # 全局 done 信号（所有 agent 都 done）
    done: jnp.ndarray          # 各 agent 的 done 信号
    action: jnp.ndarray        # 执行的动作
    value: jnp.ndarray         # Critic 输出的状态价值 V(s)
    reward: jnp.ndarray        # 获得的奖励
    log_prob: jnp.ndarray      # 动作的对数概率 log π(a|s)
    obs: jnp.ndarray           # 观察
    info: jnp.ndarray          # 额外信息
```

**形状：** `(NUM_STEPS, NUM_ACTORS_PER_TYPE, ...)` — `NUM_STEPS=64 步的轨迹`

### 5.2 TrainState（训练状态）

```python
class TrainState:
    params: dict           # 网络参数（可训练权重）
    apply_fn: function     # network.apply（前向传播函数）
    tx: optimizer          # optax 优化器
    opt_state: any         # 优化器内部状态（adam 动量等）
```

每个 agent 类型有**各自独立的 TrainState**。

---

## 六、训练流程详解

### 6.1 初始化阶段

```
1. 解析 config → 创建 MultiAgentConfig
2. 创建 MARLEnv 环境实例
3. 为每类 agent 初始化 ActorCriticRNN 网络
4. 创建 optax 优化器 (Adam + gradient clipping)
5. 初始化环境状态 → 得到初始 obs
6. 用 reshape_pytree_leading_dim 将数据分片到 N_DEVICES
```

**关键形状变换：**

```python
# 环境重置后:
obsv shape = (NUM_ENVS, num_agents, obs_dim)
    ↓ reshape_pytree_leading_dim(N_DEVICES=4)
obsv shape = (4, NUM_ENVS/4, num_agents, obs_dim)
    ↑ 分到 4 张 GPU 上，每张处理 NUM_ENVS/4 个环境
```

### 6.2 环境步 — _env_step()

**输入输出：**
```
输入: runner_state (当前所有状态)
输出: (新 runner_state, transitions 列表)

每张 GPU 上:
  对每类 agent:
    1. batchify: (N_ENVS, n_agents, obs_dim) → (N_ACTORS_PER_TYPE, obs_dim)
       ↑ 把 env 和 agent 维度打平，每个 actor 独立推理
    2. 网络前向 → pi, value
    3. pi.sample() → action (离散动作)
    4. unbatchify: (N_ACTORS, ...) → (N_ENVS, n_agents)
       ↑ 还原回 (环境, agent) 结构
  5. jax.vmap(env.step) → 在 N_ENVS 个环境上并行执行
  6. 收集 Transition: obs, action, reward, done, value, log_prob
```

**batchify 的作用：**

```
类比：
batchify 前:  
  环境 0 环境 1 环境 2 ... 环境 16383
  agent0,1  agent0,1  agent0,1  agent0,1
               ↓ batchify
batchify 后:
  actor0, actor1, actor2, ... actor(16384*5)
  (每个 actor 独立输入网络推理)
```

### 6.3 GAE 计算 — _calculate_gae()

```python
def _calculate_gae(gamma, gae_lambda, traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    # 从后往前 scan（reverse=True），逐步累积 advantage
    _, advantages = jax.lax.scan(
        _get_advantages,
        (zeros, last_val),   # 初始 gae=0, next_value=last_val
        traj_batch,
        reverse=True,        # ← 逆序遍历：从最后一步往前算
        unroll=16,
    )
    return advantages, advantages + traj_batch.value  # returns, targets
```

**GAE 公式实现：**

```math
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)           ← TD-error
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...   ← GAE
       ↑ jax.lax.scan(reverse=True) 从最后一步累积回去
```

### 6.4 IPPO 损失函数 — _loss_fn()

```python
def _loss_fn(params, init_hstate, traj_batch, gae, targets):
    # 1. 重新跑网络前向（用当前最新参数）
    _, pi, value = train_state.apply_fn(params, ...)
    log_prob = pi.log_prob(traj_batch.action)
    
    # 2. Value Loss（裁剪）
    value_pred_clipped = traj_batch.value + clip(value - traj_batch.value, -ε, +ε)
    value_loss = 0.5 * max(square(value - targets), square(value_pred_clipped - targets))
    
    # 3. Actor Loss（PPO 裁剪目标函数）
    ratio = exp(log_prob - traj_batch.log_prob)
    gae = normalize(gae)                              # advantage 标准化
    loss_actor1 = ratio * gae
    loss_actor2 = clip(ratio, 1-ε, 1+ε) * gae
    loss_actor = -min(loss_actor1, loss_actor2)
    
    # 4. Entropy Bonus（鼓励探索）
    entropy = pi.entropy()
    
    # 5. 总损失
    total_loss = loss_actor + VF_COEF * value_loss - ENT_COEF * entropy
```

### 6.5 更新循环

```
更新循环 = step(64步) → GAE计算 → epoch(4次) → minibatch(16个) → 完成
                         ↓
                  ┌──────────────────────────┐
                  │ 总计: 952 updates × 64 步 │ = 约 1e9 步
                  └──────────────────────────┘
```

```
for update in range(NUM_UPDATES):   # NUM_UPDATES = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS)
    1. _env_step × NUM_STEPS=64     → 收集 64 步轨迹
    2. _calculate_gae              → 计算 advantage 和 target
    3. for epoch in range(4):       → UPDATE_EPOCHS 轮
       3.1 随机打乱 (permutation)
       3.2 分成 16 个 minibatch    → NUM_MINIBATCHES
       3.3 for each minibatch:
           - jax.grad(_loss_fn)    → 计算梯度
           - pmean 跨 GPU 平均     → jax.lax.pmean
           - apply_gradients       → 更新网络
    4. callback → wandb 日志
```

---

## 七、多 GPU 并行：PMAP 机制

整个训练通过 `jax.pmap` 分布在多张 GPU 上：

```python
jitted_update_step = jax.jit(_update_step)

pmapped_update_step = jax.pmap(
    jitted_update_step,
    axis_name="device_batch",
    in_axes=(((0, 0, 0, 0, 0, 0), None), None, None),
    out_axes=(((0, 0, 0, 0, 0, 0), None), 0),
)
```

**in_axes 的含义**：告诉 PMAP 哪些维度按设备拆分

```
runner_state 有 6 个组件: 全部按 first axis 拆分
  (train_states, env_state, obsv, init_dones_agents, hstates, device_rng)
    ↑ 0             ↑ 0        ↑ 0    ↑ 0               ↑ 0       ↑ 0
```

**关键通信操作：**

```python
# 跨 GPU 平均梯度（all-reduce）
grads = jax.lax.pmean(grads, axis_name="device_batch")
# 跨 GPU 平均损失
total_loss = jax.lax.pmean(total_loss, axis_name="device_batch")
```

这意味着：每张 GPU 计算自己那部分数据的梯度 → all-reduce 平均 → 各卡得到**一致的参数更新** → 各卡保持参数同步。

### reshape_pytree_leading_dim

这是训练前执行的关键数据分片函数：

```python
# 假设: 16,384 个环境, 4 张 GPU
env_state_0: 形状 (16384, ...) 
    ↓ reshape
env_state_0: 形状 (4, 4096, ...)  ← 每张卡拿到 4096 个环境
```

---

## 八、三组超参数的相互作用

理解下面三组参数怎么配合，是看懂这个文件的关键：

### 8.1 数据量维

| 参数 | 值 | 作用 |
|:----|:--:|:-----|
| `NUM_ENVS` | 16384 | 并行环境数 |
| `NUM_STEPS` | 64 | 每轮收集的步数 |
| `NUM_UPDATES` | = TOTAL_TIMESTEPS / (NUM_ENVS × NUM_STEPS) | 总更新次数 |
| `N_DEVICES` | 4 | GPU 数量 |

例：`TOTAL_TIMESTEPS=2.5e8, NUM_ENVS=16384, NUM_STEPS=64`

```
NUM_UPDATES = 250,000,000 / (16,384 × 64) = 238
                        ↑ 每次更新收集这么多 timesteps = 1,048,576
```

### 8.2 并行 actor 维

| 参数 | 值 | 意义 |
|:----|:--:|:-----|
| `NUM_AGENTS_PER_TYPE` | [5,5] | MM 5 个 + EXE 5 个 |
| `NUM_ACTORS_PERTYPE` | [81920, 81920] | = 5×16384 = 每张 GPU 上 MM/EXE actor 数 |
| `NUM_ACTORS_PERTYPE / N_DEVICES` | [20480, 20480] | 每张 GPU 上的 actor 数 |

### 8.3 训练维

| 参数 | 值 | 意义 |
|:----|:--:|:-----|
| `UPDATE_EPOCHS` | 4 | 每批数据重复利用次数 |
| `NUM_MINIBATCHES` | 16 | 分成多少小批更新 |
| `MINIBATCH_SIZE` | 20480×64÷16=81920 | 每个 minibatch 的样本数 |

---

## 九、代码调用链：从 config 到环境

### 9.1 Config 解析

```python
# 1. Hydra 加载 YAML
config = load_yaml("convert_PMAP_ippo_rnn_JAXMARL_2player.yaml")

# 2. 创建 agent 配置（关键！k.lower() 把 camelCase 字段名全变成小写）
agent_configs = {
    "MarketMaking": MarketMaking_EnvironmentConfig(
        action_space="fixed_quants",
        observation_space="engineered",
        fixed_quant_value=3,
        ...
    ),
    "Execution": Execution_EnvironmentConfig(
        action_space="fixed_quants_complex", 
        observation_space="engineered",
        fixed_quant_value=10,
        ...
    )
}

# 3. 组装多 agent 配置
ma_config = MultiAgentConfig(
    number_of_agents_per_type=[5, 5],   # 从 config 读取
    dict_of_agents_configs=agent_configs,
    world_config=World_EnvironmentConfig(
        stock="600036",
        dataPath="/home/atomus/JaxMARL-HFT/data",
        ...
    )
)
```

### 9.2 环境创建

```python
from gymnax_exchange.jaxen.marl_env import MARLEnv

env = MARLEnv(
    key=init_key,
    multi_agent_config=ma_config  # ← 整个配置传进去
)
```

`MARLEnv` 是一个 JAX 多 agent 环境封装，内含：
- 订单簿引擎（共享）
- 多个 agent 类型的 reset/step 逻辑（MM 的报价行为、EXE 的执行逻辑）
- 全局状态更新

### 9.3 数据流向总图

```
  YAML Config
      │
      ▼
 OmageConf.merge + WorldConfig save/restore  [★我们修复的bug]
      │
      ▼
 make_train(config) → train(rng)
      │
      ├──▶ 创建 MultiAgentConfig → MARLEnv
      ├──▶ 创建 ActorCriticRNN × 每类 agent
      ├──▶ env.reset() → 初始 obs
      │
      ▼
  更新循环 (952 次 / 1e9 步)
      │
      ├──▶ _env_step × 64 步
      │      env.step()  ← 核心调用
      │      ↓
      │     MARLEnv.step() 内部:
      │       ├── job.process_messages(messages, orders)
      │       │   ↑ gymnax_exchange/jaxob/ — 订单簿引擎
      │       │     处理: 撤单、挂单、成交匹配、更新 L2 快照
      │       ├── mm_env.step()    → MM agent 的 reward 计算
      │       └── exec_env.step()  → EXE agent 的 reward 计算
      │
      ├──▶ _calculate_gae()
      │      ↓
      │     GAE = δ₀ + (γλ)·δ₁ + (γλ)²·δ₂ + ...
      │
      └──▶ _update_epoch × 4
             └──▶ _update_minbatch × 16
                    └──▶ _loss_fn → jax.grad → pmean → apply_gradients
```

---

## 十、所有导入模块的说明

### 10.1 来自项目内部的模块

| 导入 | 位置 | 作用 |
|:----|:-----|:-----|
| `MARLEnv` | `gymnax_exchange/jaxen/marl_env.py` | 多 agent 环境主类，统筹所有 agent 的 reset/step |
| `MultiAgentConfig` | `gymnax_exchange/jaxob/jaxob_config.py` | 最高层配置类，组合所有子配置 |
| `World_EnvironmentConfig` | 同上 | 订单簿/数据相关配置（stock, dataPath, tick_size...） |
| `MarketMaking_EnvironmentConfig` | 同上 | MM agent 参数（qty, obs_space, reward...） |
| `Execution_EnvironmentConfig` | 同上 | EXE agent 参数（task_size, action_space...） |

> `gymnax_exchange/jaxob/` 目录包含完整的订单簿引擎（JaxOrderBook），有：
> - 消息处理（挂单、撤单、修改）
> - 成交匹配算法
> - L2 快照状态更新
> - 基于 JAX 的纯函数实现，可编译加速

### 10.2 第三方库

| 库 | 用途 |
|:---|:-----|
| **jax / jax.numpy** | 核心计算：GPU 加速、jit、vmap、pmap、lax.scan |
| **flax** | 神经网络框架（nn.Module, TrainState, serialization） |
| **optax** | 优化器（Adam, gradient clipping, learning rate schedule） |
| **distrax** | 概率分布（Categorical, log_prob, entropy, sample） |
| **omegaconf** | 配置系统（OmegaConf.merge, structured configs） |
| **hydra** | YAML 配置加载 (@hydra.main decorator) |
| **wandb** | 实验追踪和日志 |
| **pandas** | 可选，训练计时保存 |

---

## 十一、入口函数和执行流程

### 11.1 正常启动: main()

```python
@hydra.main(config_path="config/rl_configs", config_name="convert_PMAP_...")
def main(config):
    # 1. 修复 bug: 保存 YAML world_config → merge → 恢复
    yaml_world_config = config.get("world_config", {})
    final_config = OmegaConf.merge(config, env_config)
    config = OmegaConf.to_container(final_config)
    config["world_config"] = yaml_world_config
    
    # 2. 包装 sweep 函数
    def sweep_fun():
        wandb.init(entity, project, tags, config, mode)
        train_fun = make_train(config)
        train_fun(rng)
        wandb.finish()
    
    # 3. 创建 wandb sweep（自动超参数搜索）
    sweep_id = wandb.sweep(...)
    wandb.agent(sweep_id, function=sweep_fun, count=500)
```

**实际我们是怎么用的：**
YAML 配置里 `WANDB_MODE: "online"`，所以直接跑了单个 sweep agent（sweep_parameters 只设了默认 LR），等价于直接训练。

### 11.2 调试入口: seperate_main()

`seperate_main()` 跟 `main()` 的唯一区别：**不做 wandb sweep**，直接创建 `make_train(config)` → `train_fun(rng)`。

### 11.3 `__name__ == "__main__"` 决定走哪个

```python
if __name__ == "__main__":
    main()    # ← 默认走 main()
```

---

## 十二、关键设计决策一览

| 决策 | 当前实现 | 考虑因素 |
|:----|:--------|:--------|
| Actor-Critic 结构 | 共享 GRU → 分开头 | 参数高效，但可能有梯度冲突 |
| RNN 类型 | GRU | 计算效率高于 LSTM，效果相近 |
| 多 agent 并行 | PMAP + Vmap | 大吞吐量，适合多 GPU |
| 参数共享 | 同类 agent 共享 | 加速训练，牺牲部分策略多样性 |
| GAE 实现 | lax.scan(reverse=True) | 完全在 GPU 上，无需 CPU 通信 |
| 数据分片 | reshape_pytree_leading_dim | 手动分片到各设备 |
| 优化器 | Adam + clip_by_global_norm | 标准 RL 配置 |
| 学习率 | 线性 Annealing | 最终收敛时精细调整 |

---

## 十三、常见困惑点

### Q: env.step 的 actions 到底是传给谁的？

```python
# actions 是一个字典或列表
actions[0] → MM agent 类型的动作 (shape: [NUM_ENVS, 5])
actions[1] → EXE agent 类型的动作 (shape: [NUM_ENVS, 5])
     ↓ env.step() 内部会分发给各自的 agent
```

### Q: 每个 GPU 上的 NUM_ENVS 怎么算的？

```python
NUM_ENVS = 16384 (config 中设置)
N_DEVICES = 4
每张 GPU = 16384 / 4 = 4096 个环境
```

### Q: `init_hstate` 的大小为什么变了两次？

```python
# 第一次: 初始化时用 NUM_ENVS 个
init_hstate = init_carry(NUM_ENVS, HIDDEN_DIM)
# 第二次: 创建 TrainState 后用 NUM_ACTORS 大小
init_hstate = init_carry(NUM_ACTORS, HIDDEN_DIM)
# 原因: 创建网络时需要看网络结构，实际训练用 actor 数量的状态
```

### Q: `NUM_UPDATES = 952` 怎么来的？

```python
TOTAL_TIMESTEPS = 1,000,000,000
NUM_ENVS = 16,384
NUM_STEPS = 64
→ 每次更新收集: 16,384 × 64 = 1,048,576 步
→ 总更新数: 1,000,000,000 / 1,048,576 = 952
```

---

## 十四、跟其他模块的接口总图

```
ippo_rnn_JAXMARL_pmap.py
         │
         │ 依赖:
         │
         ├── gymnax_exchange/jaxob/jaxob_config.py
         │    └── MultiAgentConfig, World_EnvironmentConfig,
         │        MarketMaking_EnvironmentConfig, Execution_EnvironmentConfig
         │
         ├── gymnax_exchange/jaxen/marl_env.py
         │    └── MARLEnv (reset, step, type_names, action_spaces, ...)
         │          │
         │          ├── gymnax_exchange/jaxen/base_env.py
         │          │    └── BaseLOBEnv (订单簿基础: _get_state_from_data, L2状态)
         │          │
         │          ├── gymnax_exchange/jaxen/mm_env.py
         │          │    └── MMEnv (做市商: 报价, 库存管理, spooner reward)
         │          │
         │          ├── gymnax_exchange/jaxen/exec_env.py
         │          │    └── ExecEnv (执行端: task完成, VWAP reward)
         │          │
         │          └── gymnax_exchange/jaxlobster/
         │               └── lobster_loader.py (实际数据加载)
         │
         └── config/rl_configs/convert_PMAP_ippo_rnn_JAXMARL_2player.yaml
              └── 全部超参数 + AGENT_CONFIGS + world_config
```
