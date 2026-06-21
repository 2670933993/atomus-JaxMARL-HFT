# T+1 Reward 改造说明

> 修改日期：2026-05-18
> 涉及文件：`mm_env.py`, `jaxob_config.py`, `StatesandParams.py`

---

## 一、改造背景

### 问题

原始代码的 MM reward 基于**美股做市商**生态设计，与 A 股市场有两个不可调和的不适配：

```
美股 (NASDAQ)                     A 股 (上交所/深交所)
─────────────────────────────────────────────────────
T+0: 买完立刻可以卖               T+1: 当天买入次日才能卖
Maker rebate: 挂单有返佣           无返佣，挂单反而要付佣金
做市商有正式制度                   普通股票无做市商
```

具体体现在 reward 公式中：

```python
# 当前代码 (spooner_asym_damped2)
reward = buyPnL + sellPnL     # 同一天买卖价差，T+1 下不可能
       + rebate_income        # A 股没有返佣
       + InventoryPnL_adjustment  # 这个可以保留
```

### 两个关键矛盾

| 当前假设 | A 股现实 | 影响 |
|---------|---------|------|
| 库存不变号，可日内买卖 | 买入锁定到次日，只能用旧底仓卖 | `buyPnL + sellPnL` 的 round trip 不成立 |
| 挂单有 rebate | 挂单成交也要付佣金 + 印花税 | `rebate_income` 是正向激励，实盘应该是成本 |

### 解决思路

不改变 IPPO 算法，只修改环境层的 reward 计算和库存路由逻辑。

---

## 二、改造方案

### 2.1 库存拆分

```
同一笔 inventory 拆分为两个分量：

inventory = base_inventory + intraday_buys

base_inventory (底仓)     — 上日遗留的持仓，可卖
intraday_buys (日内买入)   — 当天买入，锁定到次日
```

**路由规则：**
```
卖单成交 → base_inventory -= sell_qty
买单成交 → intraday_buys += buy_qty
        inventory = base_inventory + intraday_buys
episode 结束时 → intraday_buys → 滚入下一 episode 的底仓（通过 reset）
```

### 2.2 Reward 公式（新增 tplus1 模式）

```python
# 1. buyPnL / sellPnL — 保持不变
#    buyPnL 仍反映「买得便宜」，sellPnL 反映「卖得贵」
#    在 T+1 下，价差利润来自：底仓卖出 + 低价回补

# 2. 去掉 rebate_income
#    改为 commission_cost：对所有成交收取佣金
#    A 股佣金万 1-2.5 双边 + 卖出万 5 印花税
commission_cost = total_trade_value * commission_bps / 10000

# 3. 保留 InventoryPnL（底仓和日内仓位一起 mark to market）
#    反映库存的未实现损益——底仓和锁定仓位都承担隔夜风险

# 4. 新增：overnight_penalty
#    惩罚日终持有大量锁定仓位
overnight_penalty = overnight_penalty_lambda * jnp.abs(intraday_buys)

# 最终 reward（tplus1 模式，基于 spooner_asym_damped2）：
reward = buyPnL + sellPnL - commission_cost
       + gamma * (InventoryPnL - max(0, eta * InventoryPnL))
       - overnight_penalty
```

### 2.3 底仓的 PnL 处理

底仓（base_inventory）有成本价 `base_cost_basis = mid_price_at_reset`。

当 agent 卖出底仓时，其实已经产生了**底仓的 PnL**：

```
base_sell_PnL = (sell_price - base_cost_basis) * sell_qty / tick_size
```

这部分 PnL 已经在现有 `sellPnL` 公式中隐含了（因为 `sellPnL = (sell_price - ref_sell) * qty`），不需要额外计算。关键在于：

- `base_cost_basis` 只在 reset 时确定（= reset 时的 mid_price）
- 实际 reward 公式中的 InventoryPnL 已经包含了底仓市值的变动
- 不需要额外追踪 base_cost_basis 的变化（简化设计）

---

## 三、修改文件清单

### 1. `gymnax_exchange/jaxen/StatesandParams.py`

在 `MMEnvState` 中新增 3 个字段：

```python
@struct.dataclass
class MMEnvState():
    posted_distance_bid: int
    posted_distance_ask: int
    inventory: int
    total_PnL: float
    cash_balance: float
    # 新增 ↓
    base_inventory: int          # 底仓股数（可卖的）
    intraday_buys: int           # 日内买入股数（锁定的）
    base_cost_basis: float       # 底仓成本价（用于 PnL 计算）
```

### 2. `gymnax_exchange/jaxob/jaxob_config.py`

在 `MarketMaking_EnvironmentConfig` 中新增 3 个配置字段：

```python
initial_base_inventory: int = 0        # reset 时底仓数量（0 = 纯 T+0 模式）
commission_bps: float = 0.0            # 佣金 bps（0 = 无佣金，万1 = 1.0）
overnight_penalty_lambda: float = 0.0  # 隔夜持仓惩罚系数（0 = 无惩罚）
```

### 3. `gymnax_exchange/jaxen/mm_env.py`

| 函数 | 改动 |
|------|------|
| `reset_env()` | 初始化 `base_inventory=int(initial_base_inventory)`, `intraday_buys=0`, `base_cost_basis=mid_price` |
| `_get_state_from_data()` | 同上（默认值 0, 0, 0.0） |
| `step_env()` | state 更新时路由 inventory：卖→减 base，买→加 intraday |
| `update_state_and_get_done_and_info()` | 传入 `extras` 中的新库存分量 |
| `get_reward()` | 新增 `tplus1` reward_function 分支：去 rebate + 加 commission + 加 overnight_penalty |

---

## 四、如何使用

### YAML 配置示例

```yaml
AGENT_CONFIGS:
  MarketMaking:
    action_space: "fixed_quants"
    observation_space: "engineered"
    reward_function: "tplus1"               # ← 新增 reward 模式
    rebate_bps: 0.0                         # 去返佣
    commission_bps: 1.0                     # 新增：万1佣金
    initial_base_inventory: 500             # 新增：底仓 500 股
    overnight_penalty_lambda: 0.01          # 新增：隔夜惩罚
    # ... 其余参数不变
```

### 兼容性

| 新参数 | 默认值 | 不影响 |
|--------|:-----:|--------|
| `reward_function: "tplus1"` | 不启用（用原有 spooner 系列） | ✅ 旧 config 不写则用原 reward |
| `commission_bps: 0.0` | 0 | ✅ 无额外成本 |
| `initial_base_inventory: 0` | 0 | ✅ 无底仓，退化为 T+0 |
| `overnight_penalty_lambda: 0.0` | 0 | ✅ 无惩罚 |

**核心原则：所有新参数默认值 = 0 或等效值，已有 YAML 配置不改也能跑，行为不变。**

---

## 五、Inventory 流转图解

```
                   reset()
                      │
                      ▼
           base_inventory = 500  (配置值)
           intraday_buys = 0
           inventory = 500
                      │
             ┌────────┴────────┐
             │  step_env()     │
             ▼                 ▼
     卖单成交              买单成交
         │                    │
         ▼                    ▼
  base_inventory -= qty   intraday_buys += qty
  (不超底仓)               (锁定到次日)
         │                    │
         └────────┬───────────┘
                  ▼
          inventory = base + intraday
                  │
                  ▼
          get_reward()
    buyPnL + sellPnL - commission
    + InventoryPnL_adjustment
    - overnight_penalty * intraday_buys
                  │
                  ▼
           episode done?
       ┌──── YES ────┐ NO ──→ 继续 step
       │
       ▼
  reset() 时新的 episode:
    base_inventory += intraday_buys  (解锁)
    intraday_buys = 0
```

---

## 六、验证方法

1. **零底仓验证**：`initial_base_inventory=0`, `commission_bps=0` → 行为应与原 spooner 一致
2. **单步验证**：观察 reset 后 `base_inventory = N`，卖单成交后 base 减少，买单成交后 intraday_buys 增加
3. **reward 对比**：相同参数下，`reward_function="spooner_asym_damped2"` vs `"tplus1"`（commission=0 时应有微小差异只来自 overnight_penalty）
4. **短跑验证**：`TOTAL_TIMESTEPS=1e7` 快速看曲线是否正常

---

## 七、后续改进方向

1. **累进底仓**：当前每个 episode 独立 reset base_inventory。后续可做相邻 episode 的 inventory 传递（更接近真实连续运行）
2. **印花税细项**：当前 `commission_bps` 是统一费率。后续可拆为佣金 + 印花税（卖出万 5）
3. **涨跌停处理**：±10% 熔断时订单簿失效，需要特殊 reward 逻辑
4. **自适应底仓量**：`initial_base_inventory` 当前固定。后续可根据波动率或 mid_price 自动调整
5. **commission 分摊到步**：当前在每步 reward 中实时扣除。也可考虑 episode 结束时一次性结算（降低 reward 噪声）
