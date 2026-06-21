# OrderManager 改造说明

> 修改日期：2026-05-18
> 涉及文件：`mm_env.py`, `exec_env.py`, `jaxob_config.py`

---

## 一、改造背景

### 问题

原始代码中，每个 RL step 都执行 **cancel-all-and-repost**：

```
每步流程:
  1. getCancelMsgs() — 扫描订单簿，找到 agent 的所有当前挂单，全部撤销
  2. action_fn() — 根据 action 生成新目标订单（全部重新挂）
  3. _filter_messages() — 抵消同价位的 cancel/action（旧 `_filter_messages` 只按 price 匹配）
```

**后果：**
- **撤单率 ~100%**：即使 agent 两步之间报价完全没变，也要先撤再挂
- **丢失排队位置**：在价格-时间优先机制中，新的订单排到队尾
- **A 股合规风险**：高频撤单再挂单构成「虚假申报」
- **消息量过大**：每步发送大量冗余消息，增加 compute

### 论文作者已知此问题

在 `jaxob/JaxOrderBookArrays.py` 第 890-892 行有原始 TODO 注释：

```python
#TODO: Implement less naive version of the auto-cancel, which does not
#  cancel and re-submit in the event where the desired position
#  overlaps with the previous position.
```

但原代码只留了一个未实现的 stub `getCancelMsgs_smart()`。

---

## 二、改造方案：OrderManager

### 架构设计

```
┌──────────────────────────────────────────────────────────┐
│  RL Agent (IPPO Policy)                                  │
│  → 输出 action                                           │
├──────────────────────────────────────────────────────────┤
│  action_fn() → target_orders [price, qty, side]          │
├──────────────────────────────────────────────────────────┤
│  ▼ OrderManager (新模块，对 agent 完全透明)               │
│     1. 从订单簿查出 agent 当前挂单                        │
│     2. 比较 target vs current (side + price + qty)       │
│        - 完全相同 → 保留（不发任何消息，排队位置保留）     │
│        - 不同 → 撤旧挂新                                 │
│     3. 只发必要消息                                      │
├──────────────────────────────────────────────────────────┤
│  Cancel msgs + Action msgs → 订单簿引擎                  │
└──────────────────────────────────────────────────────────┘
```

### 核心算法

`_order_manager()` 是一个纯规则函数，在 agent 的 `get_messages()` 层级工作：

```python
def _order_manager(target_msgs, world_state, agent_params):
    # 1. 调用 getCancelMsgs 找出 agent 所有当前订单
    #    (与旧代码相同，但不直接使用)
    current_msgs = job.getCancelMsgs(...)
    
    # 2. 构建匹配矩阵 (n_current × n_target)
    #    判断条件: side相同 AND price相同 AND qty相同
    match = (same_side) & (same_price) & (same_qty) & (both_valid)
    
    # 3. 决策
    #    - 匹配的 current order → 保留（zero out cancel msg）
    #    - 匹配的 target → 跳过（zero out action msg）
    #    - 不匹配的 current → 撤销
    #    - 不匹配的 target → 新挂
    
    opt_cancel = where(~c_keep, current_msgs, zeros)
    opt_action = where(~t_skip, target_msgs, zeros)
```

**关键区别 vs 旧 `_filter_messages()`：**

| 维度 | `_filter_messages()` | `_order_manager()` |
|------|:-----:|:--------:|
| 匹配条件 | price 相同 | **side + price + qty 都相同** |
| 匹配方向 | cancel ↔ action | **cancel ↔ action 双向** |
| 保留效果 | 仍发 cancel+action（净量抵消） | **根本不发消息（排队位置保留）** |
| 排队位置 | 丢失（新老消息都发了） | **保留（不发消息=不动）** |
| 消息量 | 约 100% send rate | 仅发送 **实际改变** 的订单 |

---

## 三、修改文件清单

### 1. `gymnax_exchange/jaxob/jaxob_config.py`

在两个 agent config 类中新增字段：

```python
# MarketMaking_EnvironmentConfig
use_order_manager: bool = False  # 默认 False 保持向后兼容

# Execution_EnvironmentConfig
use_order_manager: bool = False
```

### 2. `gymnax_exchange/jaxen/mm_env.py`

- **新增** `_order_manager()` 方法（~50行）
- **修改** `get_messages()`：按 `self.cfg.use_order_manager` 分支选择流程

### 3. `gymnax_exchange/jaxen/exec_env.py`

- **新增** `_order_manager()` 方法（exec 端只操作单侧订单簿，逻辑更简洁）
- **修改** `get_messages()`：同上

---

## 四、如何使用

### 在 YAML 配置中启用

在 `AGENT_CONFIGS` 中为每个 agent 类型添加 `use_order_manager: True`：

```yaml
"AGENT_CONFIGS":
  MarketMaking:
    action_space: "fixed_quants"
    # ... 其他参数 ...
    use_order_manager: True          # ← 新增

  Execution:
    action_space: "fixed_quants_complex"
    # ... 其他参数 ...
    use_order_manager: True          # ← 新增
```

### 不修改原 YAML

默认 `False`，保留原 cancel-all-and-repost 行为，已有训练 config 无需改动。

---

## 五、预期效果

| 指标 | 改造前 | 改造后 | 改善幅度 |
|:----|:-----:|:------:|:--------:|
| 每步撤单率 | ~100% | ~10-30% | **3-10× 降低** |
| 消息总量/步 | 150 条 | 100-130 条 | 10-30% 降低 |
| 排队位置保留 | 从不 | 策略未变时保留 | **显著的排队优势** |
| A 股合规风险 | 高（虚假申报） | 大幅降低 | **合规风险可控** |
| 训练速度 | baseline | 略快（更少消息处理） | 轻微提升 |

**注意：** 这是纯工程优化，**不影响 RL agent 的学习目标或动作空间**。Agent 的输出和行为与之前完全一致，只是环境层减少了冗余的订单消息。

### 量化估算

对于 `fixed_quants` 策略（MM agent 每步输出 2 个目标订单：bid + ask）：

假设 agent 两步之间：
- 40% 概率报价不变 → 0 条 cancel + 0 条 action（排队位置完美保留）
- 30% 概率一测不变、一测变 → 1 条 cancel + 1 条 action
- 30% 概率全变 → 2 条 cancel + 2 条 action

**改造前：** 每步必发 2 cancel + 2 action = 4 条
**改造后：** 每步平均 0.3×0 + 0.3×2 + 0.3×4 = 1.8 条 → **消息量降低 55%**

---

## 六、验证方法

### 单元验证思路

1. 修改 YAML 后启动训练，观察 WandB 中的 reward 曲线
2. 如果 OrderManager 正常工作，应在**不改变 reward 均值的条件下**看到更低的 reward 方差（因排队位置更确定）
3. 第一次使用建议先用 `TOTAL_TIMESTEPS=1e8` 快速验证

### 调试输出

如需调试，可在 `mm_env.py` 或 `exec_env.py` 的 `get_messages()` 中临时添加：

```python
jax.debug.print("OrderManager | action_msgs: {}", action_msgs)
jax.debug.print("OrderManager | cancel_msgs: {}", cancel_msgs)
```

或在 `marl_env.py` 中查看 `combined_msgs` 统计。

---

## 七、后续改进方向

1. **部分数量修改**（当前版本：qty 改变就撤旧挂新）
   - 可进一步优化：同价位但 qty 改变时，只发一个 modify 消息（而不是 cancel + action）
   - 需要 LOB 引擎支持 modify 消息类型（当前只支持 new=1, cancel=2, trade=3/4）

2. **OrderManager 参数化**
   - 可考虑加一个 `order_manager_mode` 字段，支持多种策略：
     - `"strict"`：完全相同才保留（当前实现）
     - `"same_price"`：同价位就保留，只改 qty
     - `"aggressive"`：只保留完全一样的，其余清空重来

3. **Agent 感知 OrderManager**
   - 当前对 agent 透明。后续可在 observation 中加入「当前挂单状态」
   - 让 agent 知道自己的订单在队列中的位置，策略可以更优

4. **执行端的特殊优化**
   - EXE agent 的 `_order_manager()` 当前只匹配单侧订单簿
   - 可以扩展支持多价位阶梯挂单情况的高效管理

---

## 八、代码溯源

### 原始 TODO（未实现）

```python
# JaxOrderBookArrays.py:857-892
def getCancelMsgs_smart(bookside,agentID,size,side,action_msgs):
    # ... stub implementation, not finished
    return cancel_msgs

def remove_cnl_if_renewed(cancel_msgs,action_msg):
    jnp.where(cancel_msgs[:,3]==action_msg[3],)
    return cancel_msgs

# TODO: Implement less naive version of the auto-cancel, which does not
#  cancel and re-submit in the event where the desired position
#  overlaps with the previous position.
```

我们的 OrderManager 正是实现了这个 TODO 的意图——并且做得更完善（匹配 side + price + qty，不仅 price）。
