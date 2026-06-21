# 分层训练方案（Phase Training）

---

## 一、什么是分层训练

分层训练（Phase Training / Curriculum Training）是将多 agent 训练拆成多个**顺序阶段**，每个阶段只训练一部分 agent，其余 agent 使用已训练好的固定策略。

### 当前的问题

```
当前 simultaneous training:

  Step 0:   MM(随机) + EXE(随机)          ← 都从零开始
  Step 50:  MM(在学)  + EXE(在学)          ← EXE学吃单，MM学报价
  Step 100: MM(被吃)  + EXE(更会吃)        ← 军备竞赛
  Step 190: MM(-0.0005) + EXE(-0.045)     ← 都学不好
```

MM 学到的东西被 EXE 持续"破坏"——EXE 越来越会识别并吃掉 MM 的有利可图的挂单，MM 的策略在一个移动的目标上永远无法收敛。

### 分层训练

```
Phase 1: 只训 MM（EXE 用固定随机策略）
  Step 0-100:  MM 面对静态对手，学习最优做市策略
  Step 100:    MM 收敛到稳定正收益 ✅（已验证：MM Only = +0.038）

Phase 2: 冻结 MM，加入 EXE
  Step 100-200: EXE 面对一个真实的、已学会的做市商，学习最优执行策略
  Step 200:    EXE 收敛 ✅

Phase 3（可选）: 同时微调
  Step 200-250: 低 LR 同时微调两个网络，适应最终动态平衡
```

---

## 二、为什么要做分层训练

### 1. MM Only 已验证可行

单 MM 训练结果（`PMAP_GRPO_MM_only`）：

```
MM reward: -0.025 → +0.038 (正收益！)
不交易占比: 0%
动作集中在 action 4/7/9
```

说明 **MM 在 300059 上可以盈利**，之前不盈利是因为 EXE 的干扰。

### 2. EXE 面对真实 MM 才能学到真东西

当前 simultaneous 训练中 EXE 学到的是"吃一个正在学习的 MM"——这跟"吃一个成熟做市商"是完全不同的技能。冻结 MM 后，EXE 面对的是一个稳定的对手方，学到的策略才具有实际意义。

### 3. 解决非平稳问题

多 agent RL 的核心难题是**非平稳性（non-stationarity）**——一个 agent 的策略变化改变了另一个 agent 的环境。分层训练通过阶段性冻结打破了这一循环，让每个阶段的环境是平稳的。

---

## 三、可行性分析

### 技术可行性：✅ 高

| 维度 | 评估 | 说明 |
|------|------|------|
| 代码改动量 | 小 | 约 20-30 行 |
| 改动范围 | 仅训练脚本 | 不涉及环境、订单簿、数据加载 |
| 风险 | 低 | 逻辑简单，无非数值风险 |
| 验证方式 | 对比 | Phase 1 = 已跑过的 MM Only 实验 |

### 已有验证

- Phase 1 实际上已经跑过（`GRPO_MM_only.yaml`），MM = +0.038 ✅
- 关键参数（ENT_COEF=0.001, eta=0.5, CLIP_EPS=0.2）已经调优

### 潜在风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Phase 2 中 EXE 面对冻结 MM 学不到东西 | EXE 不改善 | 降低 EXE 的 reward_scaling_quo 或提高 LR |
| Phase 3 同时微调可能破坏 MM | MM 退步 | 用很低 LR（如 1e-5），小步微调 |
| 冻结参数的梯度在 jax.lax.scan 中处理不当 | JIT 编译错误 | 用 `jax.lax.stop_gradient` 或条件 apply_fn |

---

## 四、修改蓝图

### 需要修改的文件

#### 1. `gymnax_exchange/jaxrl/MARL/GRPO_ippo_rnn_JAXMARL_pmap.py`（核心改动）

**改动 1：新增训练阶段控制**

在 `make_train()` 函数开头添加阶段参数：

```python
# 在 config 解析区域添加
TRAIN_PHASE = config.get("TRAIN_PHASE", "joint")  # "mm_only" | "exe_only" | "joint"
```

**改动 2：条件性创建网络**

当前创建网络的循环是：
```python
for i, instance in enumerate(env.instance_list):
    network = ActorOnlyRNN(...)
    train_state = TrainState.create(...)
    train_states.append(train_state)
```

改为按阶段选择性创建：
```python
for i, instance in enumerate(env.instance_list):
    network = ActorOnlyRNN(...)
    train_state = TrainState.create(...)
    train_states.append(train_state)
    
    # 冻结不需要训练的 agent
    if (TRAIN_PHASE == "mm_only" and i == 1) or \
       (TRAIN_PHASE == "exe_only" and i == 0):
        # 用 optax 的 identity optimizer（不更新参数）
        train_state = train_state.replace(tx=optax.identity())
```

**改动 3：处理冻结参数**

如果 JIT 编译不允许动态选择 optimizer，可以采用更简洁的方式——在 `_update_epoch` 中跳过不需要的 agent：

```python
# 在 _update_step 的 loss 计算循环中
for i, train_state in enumerate(train_states):
    if TRAIN_PHASE == "mm_only" and i == 1:
        continue  # 跳过 EXE 更新
    if TRAIN_PHASE == "exe_only" and i == 0:
        continue  # 跳过 MM 更新
    # ... 正常 loss 计算
```

**改动 4：加载已训练参数（Phase 2 需要）**

Phase 2 需要从 Phase 1 的 checkpoint 加载 MM 的参数：

```python
# 在初始化网络后
if TRAIN_PHASE == "exe_only" and "CHECKPOINT_PATH" in config:
    with open(config["CHECKPOINT_PATH"], "rb") as f:
        pretrained_params = flax.serialization.from_bytes(
            train_states[0].params, f.read()
        )
    train_states[0] = train_states[0].replace(params=pretrained_params)
    # 用 identity optimizer 冻结 MM
    train_states[0] = train_states[0].replace(tx=optax.identity())
```

#### 2. `config/rl_configs/`（新增配置文件）

创建三个配置：

| 文件名 | 阶段 | 说明 |
|--------|------|------|
| `Phase1_MM_only.yaml` | Phase 1 | 基于 `GRPO_MM_only.yaml` |
| `Phase2_EXE_only.yaml` | Phase 2 | 新增，MM 冻结，只训 EXE |
| `Phase3_finetune.yaml` | Phase 3 | 可选，双 agent 低 LR 微调 |

Phase 2 配置示例：

```yaml
# ============================================================
# Phase 2: 冻结 MM，只训 EXE
# 前提: 需要 Phase 1 训练好的 checkpoint
# ============================================================
"TRAIN_PHASE": "exe_only"
"CHECKPOINT_PATH": "/path/to/phase1_checkpoint.pkl"
"LR": [0.0, 0.0003]    # MM 不动，EXE 高 LR
"ENT_COEF": [0.0, 0.001]
"KL_COEF": [0.0, 0.01]
# ... 其余同 GRPO_300059_v2.yaml
```

### 可能影响的模块

| 模块 | 影响 | 说明 |
|------|------|------|
| `GRPO_ippo_rnn_JAXMARL_pmap.py` | **核心改动** | 训练阶段控制 |
| `ippo_rnn_JAXMARL_pmap.py` | 无 | 不改动原始 PPO 代码 |
| `marl_env.py` | 无 | 环境逻辑不变 |
| `mm_env.py` / `exec_env.py` | 无 | reward/action 逻辑不变 |
| `jaxob/` | 无 | 订单簿逻辑不变 |
| `jaxlobster/` | 无 | 数据加载不变 |
| `convert/` | 无 | 数据转换不变 |

### 训练工作流

```
Step 1: 跑 Phase 1
  python3 GRPO_ippo_rnn_JAXMARL_pmap.py --config-name=Phase1_MM_only
  → 跑完保存 checkpoint（params.pkl）

Step 2: 跑 Phase 2
  python3 GRPO_ippo_rnn_JAXMARL_pmap.py --config-name=Phase2_EXE_only
  → 加载 Phase 1 的 checkpoint，冻结 MM，只训 EXE

Step 3（可选）: 跑 Phase 3
  python3 GRPO_ippo_rnn_JAXMARL_pmap.py --config-name=Phase3_finetune
  → 双 agent 低 LR 微调
```

---

## 五、预期效果

| 阶段 | MM reward | EXE reward | 对比当前 |
|------|-----------|-----------|---------|
| Phase 1 (MM Only) | **+0.03 ~ +0.04** | 固定 -0.06 | ✅ 已经验证 |
| Phase 2 (EXE Only) | 冻结，不变 | **优于 -0.045** | 🎯 目标 |
| Phase 3 (微调) | 保持正收益 | 继续改善 | 🎯 目标 |
| 最终 | +0.02 ~ +0.03 | -0.03 ~ -0.04 | 双 agent 都优于当前 |

---

## 六、可行性总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码改动量 | ⭐⭐⭐⭐⭐ 小 | 仅一个文件，~30 行 |
| 理论可靠性 | ⭐⭐⭐⭐⭐ 高 | 已有 MM Only 验证 |
| 风险 | ⭐⭐⭐⭐⭐ 低 | 代码逻辑简单 |
| 实验成本 | ⭐⭐⭐⭐ 中 | 需要跑三轮训练 |
