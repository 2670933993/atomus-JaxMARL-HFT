# 背景噪声 agent 方案

---

## 一、为什么要加背景 agent

### 当前的问题

```
当前模拟器:
  MM(学) ↔ EXE(学) ↔ 历史成交数据
         ↑ 零和博弈
  spread 在两者之间分配，一人赚就是另一人亏

真实市场:
  MM1   MM2   MM3 ...（竞争报价）
     ↕    ↕    ↕
  订单簿深度（来自无数市场参与者）
     ↕    ↕    ↕
  EXE1  EXE2 ...（竞争执行）
         ↓
  双方都有盈利空间，因为 spread 由 MM 之间的竞争决定
```

没有背景订单流时，MM 和 EXE 抢的是**同一块固定的 spread**。加上背景订单流：

- EXE 不再只能吃 MM 的单 → 可以吃到背景 agent 的被动单
- MM 不再独自承担全部反向风险 → 背景 agent 也在报价
- 订单簿深度动态变化 → 模拟更接近真实

---

## 二、方案设计

### 核心思路

在环境初始化时创建 N 个**零智能背景 agent**，它们每步执行固定策略：
- 有一定概率在随机价格挂一个限价单
- 有一定概率撤单（如果已有挂单）
- 不做任何学习

关键：**背景 agent 不参与 RL 训练，不需要网络、梯度、loss**。只在 marl_env.py 的 step 里额外注入消息。

### 背景 agent 行为

| 动作 | 概率 | 说明 |
|------|------|------|
| 挂一个买单 | 30% | 在 best_bid 附近随机深度挂单 |
| 挂一个卖单 | 30% | 在 best_ask 附近随机深度挂单 |
| 撤单（如果已有挂单） | 20% | 模拟撤单行为 |
| 什么都不做 | 20% | 保持当前状态 |

### 数量

| 参数 | 推荐值 | 依据 |
|------|--------|------|
| 背景 agent 数量 | **20-50**/env | 太多则淹没 RL agent 的信号，太少则没效果 |
| 每步操作数 | 同 agent 数量 | 每个背景 agent 每步做一件事 |
| 挂单数量 | 随机 1-10 股 | 跟 MM 的 fixed_quant_value 同级 |

---

## 三、改动范围

### 只改一个文件：`gymnax_exchange/jaxen/marl_env.py`

不需要修改订单簿、RL 算法、reward 函数、数据加载。

### 具体改动点

**改动 1：环境初始化（~20 行）**

```python
# 在 MARLEnv.__init__() 末尾加入
self.n_background_agents = cfg.get("n_background_agents", 0)
self.background_trader_id_start = -10000  # 与 RL agent ID 不重叠
```

**改动 2：生成背景 agent 消息（~40 行）**

```python
def _generate_background_messages(self, world_state, rng):
    """生成背景 agent 的随机订单消息。"""
    n = self.n_background_agents
    if n == 0:
        return jnp.zeros((0, 8), dtype=jnp.int32)
    
    # 每个 agent 随机选择动作
    actions = jax.random.randint(rng, (n,), 0, 4)  # 4 种动作
    
    # 挂单方向: 随机 bid/ask
    sides = jax.random.choice(rng, jnp.array([-1, 1]), shape=(n,))
    
    # 挂单价格: 围绕 best_bid/best_ask 随机偏移 1-5 tick
    prices = ...  # 基于当前盘口
    
    # 挂单数量: 1-10 股随机
    quants = jax.random.randint(rng, (n,), 1, 11)
    
    # 撤单: 从已有挂单中随机选一个取消
    # ...
    
    return messages
```

**改动 3：在 step() 中注入消息（~5 行）**

```python
# 在消息处理阶段（取消→行动→数据），在行动和数据之间插入
background_msgs = self._generate_background_messages(world_state, rng)
combined_msgs = jnp.concatenate([
    all_cancel_msgs, all_action_msgs, 
    background_msgs,  # ← 新增
    data_messages
])
```

### 总代码量

| 部分 | 行数 | 复杂度 |
|------|------|--------|
| 初始化 | ~20 | 低 |
| 消息生成 | ~50 | 中 |
| 消息注入 | ~5 | 低 |
| **总计** | **~75** | |

---

## 四、预期效果

### 短期（加背景 agent 后直接观察）

| 指标 | 当前 | 预期 |
|------|------|------|
| MM reward | -0.0007 | **改善**，因为有其他 agent 分摊风险 |
| EXE reward | -0.105 | **改善**，因为可以吃背景 agent 的被动单 |
| 订单簿深度变化 | 只减不增 | **动态平衡** |
| 训练收敛速度 | 慢 | **可能加快** |

### 长期价值

| 方向 | 意义 |
|------|------|
| 更真实的模拟环境 | 背景 agent 模拟了市场中其他参与者的随机行为 |
| 减少过拟合 | MM 不再只面对一个固定的对手模式 |
| 可直接迁移 | 背景 agent 逻辑简单，不需额外数据 |
| 性能影响小 | 背景 agent 不需要神经网络，只有数组操作 |

---

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 背景 agent 太强，淹没 RL agent 信号 | 中 | 训练效果反而变差 | 从 10 个开始试，逐步增加 |
| 性能下降（消息量增加） | 低 | 训练变慢 | 背景 agent 消息用纯 JAX 数组操作，无 Python 循环 |
| 背景 agent 行为模式太固定 | 中 | 被 MM/EXE 学会利用 | 随时间或按条件改变随机参数 |
| 与现有代码兼容性 | 低 | 编译错误 | 在 cfg 里用 n_background_agents=0 默认关闭 |

---

## 六、实验计划

| 步骤 | 内容 | 耗时 |
|------|------|------|
| 1 | 实现背景 agent 消息生成逻辑 | ~1h |
| 2 | 在 marl_env.py step 中注入 | ~0.5h |
| 3 | 用 Phase 1 配置跑一轮（n_background=10） | ~2.5h |
| 4 | 对比有无背景 agent 的 MM reward | ~0.5h |
| 5 |（可选）调参数数量/行为概率 | ~2h |

---

## 七、总评

| 维度 | 评分 | 说明 |
|------|------|------|
| 改动量 | ⭐⭐⭐⭐⭐ 小 | 一个文件 ~75 行 |
| 理论收益 | ⭐⭐⭐⭐ 高 | 打破零和博弈，更真实 |
| 风险 | ⭐⭐⭐⭐ 低 | 默认 n_background=0 时与原版完全一致 |
| 验证成本 | ⭐⭐⭐ 中 | 需要跑一轮训练验证效果 |
