# P0 改进：观测扩展 + Reward 修复

## 概述

根据 Pro 架构分析报告，对系统进行两项 P0 优先级的核心改进：

1. **观测空间扩展** — 为 MM 和 EXE agent 添加 engineered 特征
2. **Reward 函数修复** — 加入印花税并调整库存 PnL 权重

---

## A. 观测空间扩展

### A1. 做市商（MM）观测（8→11维）

**代码位置**：`mm_env.py` → `_get_obs_engineered()` 的 `fixed_steps` 分支

> ⚠️ 注意：`fixed_steps` 和 `fixed_time` 分支观测维度不同。
> `fixed_time` 分支当前为 10 维（多 delta_time/time_remaining，少 inventory_ratio），
> 本文档仅描述 `fixed_steps` 分支的改动。

**原有观测（8维）**：
```
p_bid, p_ask, spread, q_bid, q_ask, mid_price, step_counter, inventory
```

**新增强化特征（3维）**：

| 特征 | 计算方式 | 意义 |
|------|---------|------|
| `imbalance` | `(q_bid - q_ask) / (q_bid + q_ask + eps)` | 订单簿不平衡，正值=买方压力，负值=卖方压力 |
| `spread_ratio` | `spread / mid_price` | 相对价差，无量纲化后跨股票可比较 |
| `inventory_ratio` | `inventory / base_inventory` | 持仓相对于底仓的比例，归一化后的库存风险 |

**新增强化特征（3维）** → 总计 11 维

**normalization**：
| 特征 | std | 理由 |
|------|-----|------|
| imbalance | 1.0 | 值域[-1,1] |
| spread_ratio | 0.01 | A股价差通常<1% |
| inventory_ratio | 1.0 | 归一化后≈1阶 |

### A2. 执行端（EXE）观测（12→16维）

**代码位置**：`exec_env.py` → `_get_obs()` 的 `fixed_steps` 分支

**原有观测（12维）**：
```
is_sell_task, p_aggr, p_pass, spread, q_aggr, q_pass,
init_price, task_size, executed_quant, remaining_quant,
step_counter, remaining_ratio
```

**新增强化特征（4维）**：

| 特征 | 计算方式 | 意义 |
|------|---------|------|
| `vwap_deviation` | `(当前价格 - 起始价格) / 起始价格` | 从起始点开始的价格偏离，用于衡量市场变动 |
| `spread_ratio` | `spread / p_aggr` | 相对价差，无量纲化 |
| `filled_ratio` | `executed_quant / task_size` | 已完成比例，0→1 |
| `urgency` | `remaining_quant / task_size × (1 - remaining_ratio)` | 紧急性得分，越高意味着剩余仓位需要更快完成 |

**新增强化特征（4维）** → 总计 16 维

**normalization**：
| 特征 | std | 理由 |
|------|-----|------|
| vwap_deviation | 1.0 | 值域~0.1 量级 |
| spread_ratio | 0.01 | A股价差通常<1% |
| filled_ratio | 1.0 | 值域[0,1] |
| urgency | 1.0 | 值域[0,1] |

### A3. 实现方式

特征计算使用已有数据，不引入新状态跟踪。GRU hidden state 负责处理时序信息。

```python
# MM 示例
tot_vol = bid_vol_tot + ask_vol_tot
imbalance = jnp.where(tot_vol > 0, (bid_vol_tot - ask_vol_tot) / tot_vol.astype(jnp.float32), 0.0)
spread_ratio = spread / jnp.maximum(world_state.mid_price, 1)
# 注意：agent_state.inventory 是 Python int，需用 jnp.array() 包装
inv_val = jnp.array(agent_state.inventory, dtype=jnp.float32)
base_inv = jnp.maximum(jnp.array(agent_state.base_inventory, dtype=jnp.float32), 1.0)
inventory_ratio = inv_val / base_inv
```

```python
# EXE 示例
aggr_price = quote_aggr[0].astype(jnp.float32)
vwap_deviation = (aggr_price - agent_state.init_price) / jnp.maximum(jnp.abs(agent_state.init_price), 1.0)
filled_ratio = agent_state.quant_executed / jnp.maximum(agent_state.task_to_execute, 1)
urgency = remaining_quant / jnp.maximum(agent_state.task_to_execute, 1) * (1.0 - remaining_ratio + 1e-8)
```

`observation_space()` 函数维度同步更新：
- MM: 8 → 11
- EXE: 12 → 16

---

## B. Reward 函数修复

### B1. 修复内容

| 问题 | 修复 | 代码位置 |
|------|------|---------|
| 缺少印花税 | 加入 `stamp_duty = income × (stamp_duty_bps / 10000)` | `mm_env.py` → `get_reward()` |
| gamma 过低 | `inventoryPnL_gamma: 0.5 → 0.3` | `jaxob_config.py` + YAML |

### B2. 印花税（stamp duty）

A 股实际费率：

| 项目 | 费率 | 目前 | 修复后 |
|------|------|------|--------|
| 佣金（双向） | 万1~万2.5 | ❌ 关闭（默认 0，需 YAML 设 `commission_bps: 1`） | ✅ 万1（YAML 设 `commission_bps: 1`） |
| **印花税（卖出）** | **万10（千1）** | ❌ 缺失 | ✅ **万10**（默认开启） |
| 过户费（双向） | 万0.2 | ❌ 缺失（微小） | ❌ 忽略（影响<2%） |

```python
stamp_duty = income * (self.cfg.stamp_duty_bps / 10_000)
```
其中 `income` 是卖出成交总额（以元为单位），`stamp_duty_bps` 默认 10（万10）。

新的 reward 公式：
```python
reward_tplus1 = buyPnL + sellPnL 
    - commission_cost - stamp_duty
    + inventoryPnL_gamma × (...)
    - overnight_penalty_lambda × |intraday_buys|
```

### B3. gamma 调整

| 参数 | 旧值（代码默认） | 新值 | 理由 |
|------|:--------------:|:----:|:-----|
| `inventoryPnL_gamma` | 0.5 | **0.3** | Pro 分析指出原值对库存 PnL 权重偏高，适度降低到 0.3 可让策略更侧重价差收益 |

> ⚠️ 注意：`inventoryPnL_gamma` **不能在 YAML 中显式设置**。训练脚本 (`ippo_rnn_JAXMARL_pmap.py#143`) 会将所有 YAML key 做 `k.lower()`，导致 `inventoryPnL_gamma` → `inventorypnl_gamma`，不匹配 dataclass 字段名。只能通过修改 `jaxob_config.py` 中的 dataclass 默认值来调整。

### B4. 配置修改

`jaxob_config.py` 新增字段：

```python
stamp_duty_bps: float = 10.0  # A股印花税 万10, 0 表示关闭
inventoryPnL_gamma: float = 0.3  # 从 0.5 降低
```

所有 YAML 配置文件新增（仅 stamp_duty_bps，inventoryPnL_gamma 由 dataclass 默认值提供）：
```yaml
  MarketMaking:
    ...
    stamp_duty_bps: 10
```

---

## C. 修改文件清单

| 文件 | 改动类型 |
|------|---------|
| `mm_env.py` | `_get_obs_engineered()`: +3 features |
| `mm_env.py` | `observation_space()`: 8→11 |
| `mm_env.py` | `get_reward()`: +stamp_duty |
| `mm_env.py` | extras dict: +stamp_duty 日志 |
| `exec_env.py` | `_get_obs()`: +4 features |
| `exec_env.py` | `observation_space()`: 12→16 |
| `jaxob_config.py` | `MarketMaker_EnvironmentConfig`: +stamp_duty_bps, gamma 0.3 |
| 3× YAML 配置 | 添加 `stamp_duty_bps: 10`；`TOTAL_TIMESTEPS`: 5e8→2.5e8 |

---

## D. 预期影响

| 改进 | 预期收益 | 信度 |
|------|---------|------|
| MM 观测扩展 | MM 能感知订单簿压力和相对持仓风险 | 中等（信息增益明确） |
| EXE 观测扩展 | EXE 能感知紧迫程度和相对成本 | 中等-高 |
| 印花税 | reward 更接近 A 股实际成本结构 | 高（公式修正） |
| gamma 0.3 | 库存管理策略能学到更多 | 中等（需要实验验证） |

## E. 下一步

同步更新到实验机后，建议重新跑 baseline 对照实验：
- 先用旧参数（sleek config）跑一次确认可复现
- 再用新参数跑一次看观测+reward 联合改善效果
