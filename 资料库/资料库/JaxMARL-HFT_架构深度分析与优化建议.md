# JaxMARL-HFT 架构深度分析与优化建议

> 审查日期: 2026-05-19
> 审查范围: 完整代码库 (环境层 / 算法层 / 配置层)
> 模型: DeepSeek V4 Pro

---

## 目录
1. [A. 网络架构 (GRU)](#a-网络架构-gru)
2. [B. Reward 函数](#b-reward-函数)
3. [C. 超参数体系](#c-超参数体系)
4. [D. 架构设计质量](#d-架构设计质量)
5. [E. 代码质量](#e-代码质量)
6. [F. 下一步方向](#f-下一步方向)
7. [总结与优先级矩阵](#总结与优先级矩阵)

---

## A. 网络架构 (GRU)

### A1. FC128→GRU128→FC 是否过于简单？

**结论：在当前的 observation 信息量下，架构足够但有余量。**

当前 ActorCriticRNN 结构：
```
obs(8→10维) → Dense(128, relu) → GRUCell(128) → Dense(128, relu) → Actor(Dense→action_dim) / Critic(Dense→1)
```

**分析：**
- 输入观测维度很小（MM 7维、EXE 8维），FC128→GRU128 的表征能力已经远超输入信息量，不会被输入维度限制
- GRU128 隐藏状态 = 16,512 参数，对于捕捉几十步内的市场微观结构，这是充足的
- 瓶颈不在网络深度，而在**输入信息量不足以让网络学到更复杂的策略**

**具体评估：**

| 维度 | 当前状态 | 评分 |
|------|---------|------|
| 输入→FC 映射 | obs_dim(8) → 128, 共有 1,152 参数 | 充足 |
| FC→GRU 映射 | 输入128 + 隐藏128 → 128, 共 32,896 参数 | 充足 |
| GRU→Actor/Critic | 128→128→act_dim, 128→128→1 | 标准 |
| 总参数量 | ~50K（单 agent 类型） | 对于 8 维obs 偏大 |

**建议**：
- **P2** - 在网络中增加对关键观测的手工特征增强层（如 spread/mid_price ratio in ticks），而非仅靠 relu 学习非线性
- **P2** - 考虑在 GRU 前增加 LayerNorm，稳定隐藏状态（当前 GRU 无归一化）
- 网络不是瓶颈，不需要盲目加深

### A2. GRU 256 vs RNN vs Transformer

**GRU 256 最可能改进什么？**

GRU hidden_dim 128 → 256 带来：

| 改进点 | 预期效果 | 概率 |
|--------|---------|------|
| 更多记忆容量 | 捕捉更长周期的市场模式（如 20+ 步前的 order flow 对当前影响） | 中等 |
| 更丰富的内部表征 | 多个 agent 同一类型的策略差异化 | 中低 |
| 增加过拟合风险 | 仅一天数据训练（~4000 窗口），参数量翻倍 | 中等 |
| 训练速度下降 | GRU 计算量 ~O(d²)，256²/128² = 4x 每个 step | 已观察到 |

**sleek (128) vs GRU256 对照实验的预期结果：**
- 若 EXE reward 在 GRU256 下提升，说明 EXE 需要更长记忆窗口
- 若 MM reward 无明显变化，说明 MM 信息大多包含在最近几步内
- 大概率：MM 持平，EXE 小幅改善（0.05-0.2），但训练速度下降明显

**RNN vs Transformer 适合这个场景吗？**

| 架构 | 优势 | 劣势 | 适合度 |
|------|------|------|--------|
| GRU (当前) | 参数量小、JAX 优化好、序列长度短时效率高 | 长程依赖较弱 | ⭐⭐⭐⭐⭐ |
| LSTM | 比 GRU 记忆稍好 | 参数更多（~1.3x）、JIT trace 稍慢 | ⭐⭐⭐⭐ |
| Transformer (纯) | 全局注意力、并行化 | 需要位置编码、参数多、episode 64-step 太短不划算 | ⭐⭐ |
| 1D Conv + GRU | 局部模式提取 + 序列建模 | 实现复杂度增加 | ⭐⭐⭐ |

**结论：GRU 是当前条件下最优选择。** episode 仅 64 steps，Transformer 的自注意力优势无法发挥，反而引入过多参数和训练不稳定。若将来 episode 扩展到 500+ steps 或使用 tick-level 消息级观测，可考虑 Transformer。

### A3. 8维 engineered observation + GRU128 的表征瓶颈

**MM 观测 (fixed_steps 模式, 8维):**
```
p_bid, p_ask, spread, q_bid, q_ask, mid_price, step_counter, inventory
```

**EXE 观测 (fixed_steps 模式, 8维):**
```
is_sell_task, p_aggr, p_pass, spread, q_aggr, q_pass, init_price, task_size, 
executed_quant, remaining_quant, step_counter, remaining_ratio
```

**严重缺失的信息：**

| 缺失信息 | 重要性 | 原因 |
|---------|--------|------|
| **订单簿不平衡 (order book imbalance)** | ⭐⭐⭐⭐⭐ | bid_vol/(bid_vol+ask_vol) 是短期价格方向的最强信号 |
| **近期已成交 trades 的信息** | ⭐⭐⭐⭐⭐ | 成交方向、成交量的统计，论文 [Cont et al. 2014] 证明其预测力 |
| **价差变化趋势 (spread delta)** | ⭐⭐⭐⭐ | spread_t - spread_{t-1} 反映流动性变化方向 |
| **中间价变化趋势** | ⭐⭐⭐⭐ | 过去 N 步的 mid_price 变化率 |
| **波动率估计** | ⭐⭐⭐⭐ | rolling std of mid_price returns |
| **自身订单在队列中的位置估计** | ⭐⭐⭐ | 知道自己是不是排在最前面 |
| **多档深度分布** | ⭐⭐⭐ | Level 2-5 的累计量能反映隐藏流动性 |
| **对手方 agent 的已知信息** | ⭐⭐ | 其他 agent 的近期行为 |

**隐式信息（通过 GRU 记忆间接获得的）：**
- mid_price 的历史轨迹（通过 GRU hidden state）
- 过去成交节奏（通过 GRU hidden state）
- 但 GRU 128 只有 ~16K 内部状态，信息压缩损失大

**建议：**
- **P0** - 扩展观测至 15-20 维：增加 imbalance、spread_delta、mid_price_return(最近3步)、vwap_deviation
- **P1** - 对于 [10,10] agents 实验，增加 self 专属特征（自身订单队列位置）
- **P1** - QOL 工程改进：用 RunningMeanStd wrapper 替代手动 normalization

---

## B. Reward 函数

### B1. T+1 reward 数学正确性与参数合理性

**T+1 reward 公式 (mm_env.py:2410-2413):**
```python
reward_tplus1 = buyPnL + sellPnL - commission_cost 
    + inventoryPnL_gamma*(InventoryPnL - max(0, inventoryPnL_eta*InventoryPnL)) 
    - overnight_penalty_lambda * |intraday_buys|
```

**逐步分析：**

**1. buyPnL + sellPnL（交易盈亏）** ✅ 正确
```
buyPnL = Σ(ref_price - trade_price)/tick_size * |qty|
sellPnL = Σ(trade_price - ref_price)/tick_size * |qty|
```
- ref_price 默认使用 mid_avg（区间均价），合理
- 已按 tick_size 归一化，单位一致

**2. commission_cost = (income + outgoing) * commission_bps/10000** ✅ 正确
- commission_bps=1.0 代表万1佣金
- **但这里有个问题**：A 股实际佣金约万1（0.01%），但还有印花税（卖出收千1=万10）。当前只收了双向佣金，**漏了印花税**。
- 建议：`total_txn_cost = commission_bps/10000 * (income+outgoing) + stamp_duty_bps/10000 * sell_volume`

**3. inventoryPnL_gamma * (InventoryPnL - max(0, η*InventoryPnL))** 
- InventoryPnL = inventory × (mid_price_end - mid_price_start) / tick_size
- 当 InventoryPnL > 0（有利方向）：扣减 η=0.6 的部分，即保留 40%
- 当 InventoryPnL < 0（不利方向）：全额保留（惩罚全部不利变动）
- gamma=0.1 再整体缩放到 10%
- **设计意图**：降低库存方向性赌博的激励，鼓励纯 spread 捕获利

- ✅ **数学正确性**：正确
- ⚠️ **参数合理性**：gamma=0.1 非常激进
  - 意味着库存 PnL 仅贡献 reward 的 10%
  - 对于做市商，库存管理是核心技能，过低 gamma 会导致 agent 学会"只要 spread 够大就报价，不管库存风险"
  - 对比论文 Spooner (2023)：标准 dampening 参数 η=0.5，gamma=1.0（不缩放整个项）

**4. overnight_penalty_lambda * |intraday_buys|** 
- 惩罚当日买入但无法卖出的仓位（T+1 锁定）
- overnight_penalty_lambda=0.01
- **问题**：惩罚与持仓规模成正比，但没有考虑持仓的时间价值
- **改进建议**：`penalty = λ * |intraday_buys| * (mid_price/tick_size) * risk_free_rate * days_to_settle`
  - 或者至少按比例归一化：`λ * |intraday_buys| / base_inventory`

**当前训练配置的 commission_bps=1.0 参数评估：**

| 项目 | 当前值 | A股实际 | 差异 |
|------|--------|---------|------|
| 佣金(双向) | 万1 | 万1-万2.5 | 合理 |
| 印花税(卖出) | 无 | 万10 | **严重缺失** |
| 过户费 | 无 | 万0.2 | 微小 |

**结论**：
- 公式数学正确 ✅
- 参数调整方向：
  - **P0** - 加入印花税（至少卖出端万10）
  - **P1** - gamma=0.1 过于压制 inventory PnL，建议 0.3-0.5
  - **P1** - overnight_penalty 需要按仓位价值归一化

### B2. EXE "normal" reward 的可优化空间

**当前 normal reward:**
```python
reward = advantage + reward_lambda * drift
advantage = direction_switch * (QP_agent - P_vwap * agentQuant)  # 执行价格 vs VWAP 的偏离
drift = direction_switch * agentQuant * (P_vwap - init_price/tick_size)  # VWAP vs 初始价的偏离
```

**分析：**
- ✅ advantage 项：对标 VWAP，学术标准。买入低于 VWAP → 正 advantage
- ✅ drift 项：按方向调整，买入时价格上涨 → 正 drift（可理解为"运气"成分）
- reward_lambda=0.0 → **当前只考虑 VWAP 偏离，不考虑市场价格方向**

**可优化空间：**

| 优化方向 | 收益 | 复杂度 | 优先级 |
|---------|------|--------|--------|
| **加入 VWAP 时间偏离惩罚** | 高 | 低 | P0 |
| 加入 Implementation Shortfall | 高 | 中 | P1 |
| 加入 urgency 惩罚（剩余时间越少越激进） | 中 | 低 | P1 |
| 加入 spread cost 感知 | 中 | 中 | P2 |
| 动态 reward_lambda（episode 前期大λ，后期小λ） | 中 | 中 | P2 |

**VWAP 偏离惩罚建议：**
```python
# 当前
reward = advantage  # reward_lambda=0

# 建议改进1：加入 VWAP 偏离惩罚
vwap_deviation_penalty = alpha * abs(P_exec_avg - P_vwap) / P_vwap
reward = advantage - vwap_deviation_penalty

# 建议改进2：流动性成本感知
spread_cost = agentQuant * spread / (2 * mid_price)
reward = advantage - beta * spread_cost

# 建议改进3：时间加权 urgency
time_weight = 1 + urgency_lambda * (1 - remaining_ratio)
reward = time_weight * advantage
```

**为什么 sleek (ENT=0.05, EXE=-0.629) 比 divine/ancient performance 好？**
- 更高 entropy (0.05) 让 EXE 动作分布更均匀，避免了"只用一种动作"的局部最优
- action1_10 (M×5) 从 46%→12% 说明 agent 不再过度依赖单一动作
- 这说明当前 EXE reward 的信号不够丰富，agent 容易 collapse 到简单策略

### B3. Inventory PnL 衰减参数分析 (η=0.6, γ=0.1)

```
reward_inventory_component = γ * (InventoryPnL - max(0, η*InventoryPnL))
                            = 0.1 * (InventoryPnL - max(0, 0.6*InventoryPnL))
```

**场景分析：**

| 场景 | InventoryPnL | 有效贡献 | 说明 |
|------|-------------|---------|------|
| 库存增值 +100 | +100 | 0.1×(100-60) = 4 | 正向只给 4% 权重 |
| 库存贬值 -100 | -100 | 0.1×(-100-0) = -10 | 负向给 10% 权重 |
| 库存不变 | 0 | 0 | 不贡献 |

**非对称比 = 2.5:1（负向惩罚 : 正向奖励）**

**效果：**
- ✅ 强激励 zero-inventory：agent 被强烈惩罚持有亏损库存
- ✅ 减少方向性赌博：agent 不会为了库存增值而冒险
- ❌ 可能过于保守：agent 可能学会"只做立即能平仓的交易，放弃有利可图的库存机会"
- ❌ **gamma=0.1 使整个库存项近乎消失**：±10 tick 的 InventoryPnL 大约贡献 ±0.4 ~ ±1.0

**建议：**
- **P0** - 提高 gamma 到 0.3-0.5，让库存管理在 reward 中有实际权重
- **P1** - 考虑分阶段策略：前期 gamma 低（学会 spread capture），后期 gamma 高（学会库存管理）
- **P1** - 增加库存上限的硬约束 reward penalty（超过阈值给大负 reward）

---

## C. 超参数体系

### C1. GAMMA 差异的合理性

| Agent | GAMMA | 半衰期 (steps) | 含义 |
|-------|-------|---------------|------|
| MM | 0.999999999 | ~6.9×10^8 | 几乎无折扣，视所有未来 reward 等同 |
| EXE | 0.99 | ~69 | 约 69 步外的 reward 权重降到 50% |

**MM 的 gamma≈1 的合理性** ✅：
- 做市商每个 step 的 spread capture 都是独立的利润来源
- 不存在明显的"牺牲现在换未来"的 trade-off
- Gamma≈1 让 agent 对所有 step 的 reward 同等重视

**EXE 的 gamma=0.99 的合理性** ✅：
- 执行任务有明确终点（task 完成），75% 的 reward 集中在最后几步
- Gamma=0.99 约 69 步半衰期，对于一个 64-step episode 来说，最后一步的折扣因子 = 0.99^63 ≈ 0.53
- 这合理，因为最后的 forced unwind 会产生大额 reward/penalty

**但存在一个问题**：
```python
# 当前代码只使用 global_done 做 bootstrap
delta = reward + gamma * next_value * (1 - global_done) - value
```
- MM 和 EXE 共享同一个 episode done 信号，但各自有不同的 gamma
- EXE agent 可能在步 30 就完成任务（is_terminal = task done），但 world 还没结束
- 此时 EXE 的 done 被设为 True（返回 0 obs），但 `global_done` 还是 False
- **这可能造成 credit assignment 问题**：EXE 做完任务后的步骤，其 value bootstrap 仍使用 world 的 done 信号而非 agent 自己的 done

**建议**：
- **P1** - 为每个 agent 类型独立计算 advantage，使用各自的 done flag 和 gamma
- 当前实现虽非最优但影响有限（EXE 做完后 reward=0，不影响训练），可暂缓

### C2. ENT_COEF 退火策略建议

**当前状态：**
- MM: ENT=0.01, EXE: ENT=0.01 (divine/ancient) → ENT=0.05 (sleek)
- 固定值，无退火

**观察：**
- sleek (ENT=0.05) 的 EXE reward(-0.629) 远好于 ENT=0.01(-1.214, -1.424)
- 说明高 entropy 有助于 EXE 探索更好的执行策略

**建议的退火策略：**

```
Phase 1 (0-30% 总步数): ENT_COEF = 初始值 × 2
  - 广泛探索动作空间
Phase 2 (30-70%): ENT_COEF = 初始值
  - 平衡探索与利用
Phase 3 (70-100%): ENT_COEF 线性衰减到 初始值 × 0.1
  - 收敛到确定性策略
```

**具体实现**：
```python
def entropy_schedule(count, total_updates, init_ent, min_ent=0.001):
    progress = count / total_updates
    if progress < 0.3:
        return init_ent * 2.0  # 高探索
    elif progress < 0.7:
        return init_ent  # 正常
    else:
        # 线性衰减
        frac = (progress - 0.7) / 0.3
        return init_ent * (1 - frac) + min_ent * frac
```

**对于 [10,10] agents 实验**：
- **P0** - 更高的 agent 数量意味着更大的策略空间，需要更积极的熵退火
- 建议初始 ENT 从 0.05 开始（而非 0.01），再退火到 0.005

### C3. [10,10] agents 的参数调整方向

| 参数 | [5,5] 当前 | [10,10] 建议 | 理由 |
|------|-----------|-------------|------|
| ENT_COEF (MM) | 0.01 | 0.05→退火 | 更多 agent 需要更广泛探索 |
| ENT_COEF (EXE) | 0.05 | 0.05→退火 | 维持高探索，避免 collapse |
| LR (both) | 2.4e-4 | 1.5e-4 | 更多 agent = 更多梯度 noise，需降低 LR |
| NUM_MINIBATCHES | 4 | 8 | 更多数据需要更好的采样 |
| GAE_LAMBDA (both) | 0.85/0.95 | 0.95/0.95 | 更多 agent 间交互需要更长 credit chain |
| NUM_STEPS (rollout) | 64 | 128 | 更多 agent 收敛更慢，需更长 rollout |
| CLIP_EPS | 0.2 | 0.15 | 更保守的更新 |

**注意**：[10,10] 将订单消息量从 60+5×4+5×4=100 msg/step 提升到 60+10×4+10×4=140 msg/step，+40% 计算量。

---

## D. 架构设计质量

### D1. OrderManager 设计质量

**设计思路**：避免"全撤-全挂"的虚假申报问题，仅对变化的订单进行撤单和重新挂单。

**详细评估**：

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | ⭐⭐⭐⭐⭐ | 正确比较 target vs current 在 price/qty/side 维度 |
| 效率提升 | ⭐⭐⭐⭐ | 理想情况下减少 50-80% 的消息量 |
| JAX 兼容性 | ⭐⭐⭐⭐ | 使用纯 JAX 操作（矩阵 broadcast + where） |
| 边界处理 | ⭐⭐⭐⭐ | 处理了 zero-price 过滤、多订单匹配 |

**MM OrderManager (mm_env.py:1881-1955):**
```python
# 核心逻辑
match = same_side & same_price & same_qty & both_valid  # (n_cancel, n_action)
c_keep = jnp.any(match, axis=1)  # 现有订单被目标匹配 → 保留
t_skip = jnp.any(match, axis=0)  # 目标被现有订单匹配 → 跳过
opt_cancel = jnp.where(~c_keep[:, None], all_current, 0)  # 只在确实变化时撤单
opt_action = jnp.where(~t_skip[:, None], target_msgs, 0)  # 只在确实变化时挂单
```

✅ 设计质量高。正确实现了"changed order only"的语义。

**小问题**：
- 1-to-many 匹配 (同一现有订单匹配多个 target) 未处理 → 但固定 quant 动作空间下不会发生
- 依赖 `job.getCancelMsgs` 提供的 size 参数来限制返回订单数

### D2. Observation 8维 engineered 空间是否充足

**结论：不充足。这是当前架构最大的瓶颈之一。**

已在 A3 节详细分析。核心问题是：
1. 缺少订单簿不平衡 (Order Book Imbalance)
2. 缺少成交流信息
3. 缺少趋势/动量特征
4. 缺少自身订单状态

**P0 优先级改进建议（15-18 维 engineered obs）：**

**MM 扩展观测：**
```
原有 7 维（fixed_steps）:
p_bid, p_ask, spread, q_bid, q_ask, mid_price, inventory

新增（+7 维→14 维）:
imbalance = (q_bid - q_ask) / (q_bid + q_ask + 1)     # 订单簿不平衡
spread_ratio = spread / mid_price                       # 相对价差
mid_price_ret_1 = (mid_t - mid_{t-1}) / mid_{t-1}      # 中间价变动
mid_price_ret_3 = (mid_t - mid_{t-3}) / mid_{t-3}      # 3步趋势
volatility_5 = std(mid_returns_{t-5:t})                 # 5步波动率
trade_imb_ratio = (buy_vol - sell_vol) / total_vol      # 最近成交不平衡
self_queue_position = 距最优价的队列深度估计             # 自身订单位置
```

**EXE 扩展观测：**
```
原有 8 维（fixed_steps）→ 15 维:
原有 + vwap_deviation, spread_ratio, trade_pressure, 
      urgency_score, participation_rate
```

### D3. 数据加载 Pipeline 设计合理性

**架构：**
```
LOBSTER CSV → LoadLOBSTER_resample → pre-compute window starts/ends/init_states 
→ pickle cache → LoadedEnvParams → jax.lax.dynamic_slice_in_dim (per step)
```

**评估**：

| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | ⭐⭐⭐⭐⭐ | 正确处理 LOBSTER 7 文件格式（message + orderbook） |
| 效率 | ⭐⭐⭐⭐ | pickle 预计算窗口初始化状态，避免重复计算 |
| 可扩展性 | ⭐⭐⭐ | 当前单股票，多股票需改为嵌套 dict |
| 数据质量 | ⭐⭐⭐ | 仅用 LOBSTER 重建，无原始逐笔委托数据 |

**问题：**
1. **window 预计算策略**：`start_resolution=64` 为每个窗口预计算 InitState → 约 4000+ 个窗口 → pickle 文件可能很大
2. **单日数据限制**：目前仅使用一天 LOBSTER 数据，跨日连续性不存在
3. **A 股数据转换**：从逐笔成交合成 LOB 时，深度分布是模拟的而非真实

**建议**：
- **P1** - 增加跨日数据支持（多天数据拼接训练）
- **P2** - 考虑 lazy loading 大数据集（当前全部加载到 GPU 内存）
- **P2** - 对于 A 股，优先获取逐笔委托（Level-2）数据

---

## E. 代码质量

### E1. mm_env.py 3569行的重构方向

**当前问题：**
- 单个文件包含：reward计算、observation构造、action消息生成、OrderManager、episode end处理、消息过滤、tokenizer、最佳价填充
- 职责混乱，难以测试和维护

**建议的重构方向：**

```
mm_env.py → 拆分为：
├── mm_env.py (核心环境逻辑, ~800行)
│   ├── MarketMakingAgent.__init__, reset_env, step_env, is_terminal
│   └── get_messages, update_state_and_get_done_and_info
├── mm_actions.py (动作消息生成, ~600行)              [P1]
│   ├── _getActionMsgs_fixedQuant
│   ├── _getActionMsgs_spread_skew
│   └── 其他 action space 实现
├── mm_rewards.py (reward 函数族, ~500行)             [P1]
│   ├── get_reward (主函数)
│   ├── _compute_spooner_rewards
│   ├── _compute_tplus1_reward
│   └── _compute_complex_reward
├── mm_observations.py (观测构造, ~400行)              [P1]
│   ├── get_observation
│   ├── _get_obs_engineered
│   ├── _get_obs_basic
│   └── _get_obs_msg_new_tokenizer
├── mm_order_manager.py (订单管理, ~200行)             [P1]
│   ├── _order_manager
│   ├── _filter_messages
│   └── _filter_messages (如有独特逻辑)
└── mm_utils.py (辅助函数, ~300行)                     [P2]
    ├── _ffill_best_prices
    ├── _extract_agent_trade_stats
    ├── locate_type_4
    └── normalize_obs
```

**exec_env.py 同理拆分为类似的模块结构。**

**优先级**：P1（不影响功能但影响维护效率）

### E2. JAX JIT 编译热点

**识别到的 JIT 热点：**

| 热点 | 位置 | 复编译频率 | 影响 |
|------|------|-----------|------|
| `marl_env.step_env` | marl_env.py | 首次+resharding | 最大开销(~数秒) |
| `scan_through_entire_array_save_bidask` | JaxOrderBookArrays | 取决于消息量 | 核心计算 |
| `_env_step` (rollout loop) | ippo_rnn_JAXMARL_pmap.py | 首次 | 中等 |
| `_update_minbatch` (PPO update) | ippo_rnn_JAXMARL_pmap.py | 首次 | 中等 |

**减少 JIT 复编译的建议：**
- ✅ `jax_disable_jit = False`（已正确设置）
- ✅ `static_argnums` 已标注关键函数
- ⚠️ `step_env` 中 `scan_through_entire_array_save_bidask` 被调用的 n_data_msg_per_step 是 JIT 时的常量，改变此参数会触发 re-JIT
- ⚠️ pmap 改变 NUM_ENVS 或 N_DEVICES 会触发全部 re-JIT
- 建议增加 `jax.config.update("jax_log_compiles", True)` 在实验初期诊断

**P2 优化方向：**
- 将 `n_data_msg_per_step` 设为固定上限，不足部分用 padding
- 考虑使用 `jax.lax.cond` 减少 branch→减少 trace 数量

### E3. 训练脚本模块化程度

**当前训练脚本 (ippo_rnn_JAXMARL_pmap.py) 模块化评估：**

| 维度 | 评分 | 说明 |
|------|------|------|
| 网络定义 | ⭐⭐⭐⭐ | ScannedRNN + ActorCriticRNN 独立类 |
| 训练循环 | ⭐⭐⭐ | make_train 闭包方式灵活但有大量嵌套 |
| 配置集成 | ⭐⭐⭐ | OmegaConf+Hydra 但手工 merge 有 bug 记录 |
| 日志/监控 | ⭐⭐⭐ | WandB 回调，但 action_distribution 手工计算 |
| 评估分离 | ⭐⭐⭐⭐ | 独立的 eval_env + eval_timeperiod |
| 代码复用 | ⭐⭐ | pmap 和非 pmap 版本代码高度重复 |

**具体问题：**

1. **ippo_rnn_JAXMARL.py vs ippo_rnn_JAXMARL_pmap.py**：大量代码重复
2. **make_train 闭包**：状态管理在闭包内部，难以单独测试
3. **callback 函数**：内嵌在 train() 内部，缺少模块化
4. **reshape_pytree_leading_dim**：通用工具函数却内联定义

**建议**：
- **P1** - 统一 pmap 和非 pmap 版本，通过 `--n_devices` 参数控制
- **P1** - 将 `_update_step`, `_env_step`, `_calculate_gae`, `_update_epoch` 提取为独立函数
- **P2** - 创建 `metrics.py` 统一管理 logging/metrics

---

## F. 下一步方向

### F1. 多股票迁移学习路径

**路径建议（由易到难）：**

**Phase 1: 多天同股票 (P0, 1-2天)**
```
目标：验证时序泛化能力
方法：
1. 用连续的 5-10 个交易日数据训练（每天随机采样窗口）
2. 在未来的 1-2 天数据上评估
3. 观察：MM spread capture 在高低波动日是否稳定
```

**Phase 2: 同类股票迁移 (P1, 1-2周)**
```
目标：验证跨股票泛化
方法：
1. 在大型银行股（601398 工行）训练
2. 转移到同类股票（601939 建行, 600036 招行）
3. 使用 finetune（冻结 GRU，只训练最后 FC 层）
4. 关键：需要将价格归一化（除以股票均价）
```

**Phase 3: 通用做市 + 条件化 (P2, 2-4周)**
```
目标：一个模型适配多股票
方法：
1. 在观测中加入股票特异性特征：
   - avg_daily_volume（日均成交量）
   - typical_spread（典型价差）
   - volatility_class（波动率分档）
2. 多股票联合训练
3. 使用 HyperNet 或 FiLM 层进行条件化
```

**关键挑战**：
- 不同股票的 tick_size 不同（A 股 tick_size 随价格区间变化）
- 不同股票的流动性特征完全不同
- 需要确保观测归一化在不同股票间可比

### F2. [10,10] agents 预期现象

**预期现象：**

| 现象 | 概率 | 原因 |
|------|------|------|
| MM 总 reward 下降但 per-agent spread 稳定 | 高 | 竞争加剧，单 agent 捕获的 spread 减少 |
| 更强的 spread 压缩（bid-ask 价差收窄） | 高 | 更多 MM agent 在同一 level 报价 |
| EXE 成本降低（更好执行价格） | 高 | 更多流动性提供者→更小 market impact |
| 某些 agent 会"消亡"（inventory=0, PnL~0） | 中高 | 竞争中弱势 agent 被挤出 |
| 训练收敛更慢 | 高 | 20 agent 的信用分配问题更复杂 |
| 出现新策略（如专门的价格层次策略） | 中 | 更多 agent 可能有隐性角色分化 |
| ENT collapse 更严重 | 中 | 更多 agent 更容易 collapse 到相同动作 |

**监控重点**：
1. 每个 agent 的 action distribution 差异度 (KL divergence between agent action distributions)
2. per-agent PnL 排序稳定性（是否有 agent 始终排前列）
3. spread 时间序列（是否比 [5,5] 收窄）
4. 订单深度分布（是否更多 agent 在不同层次报价）

**可能需要的干预**：
- 如果所有 agent 收敛到相同策略 → 增加 ENT_COEF 或加入多样性奖励
- 如果有 agent 完全不交易 → 加入最低交易量惩罚
- 如果 spread 过度压缩 (MM 全亏损) → 增加 MM 数量惩罚或降低 EXE 数量

### F3. 实际 A 股 Deployment 需要的改动

**改动清单（按优先级）：**

| 序号 | 改动 | 类别 | 优先级 | 工作量 |
|------|------|------|--------|--------|
| 1 | **实时数据接入**：替换 LOBSTER 回放为 WebSocket/API 实时流 | 数据 | P0 | 2-3周 |
| 2 | **交易接口对接**：CTP/XTrem 等柜台系统接口 | 执行 | P0 | 3-4周 |
| 3 | **Tick-to-trade 延迟**：模型推理延迟 < 50μs（FPGA/GPU 直通） | 性能 | P0 | 4-6周 |
| 4 | **风控层**：持仓上限、单日亏损上限、撤单率监控 | 风控 | P0 | 1-2周 |
| 5 | **涨跌停处理**：A 股 ±10% 限制，停牌处理 | 逻辑 | P1 | 3-5天 |
| 6 | **T+1 约束硬编码**：当日买入不可卖出 | 逻辑 | P1 | 已部分实现 |
| 7 | **最小报价单位动态调整**：不同价格区间不同 tick_size | 逻辑 | P1 | 2-3天 |
| 8 | **集合竞价处理**：9:15-9:25 连续竞价不同机制 | 逻辑 | P2 | 1周 |
| 9 | **交易成本完整建模**：佣金+印花税+过户费 | 逻辑 | P0 | 1天 |
| 10 | **监控与告警系统**：实时 PnL 监控、异常检测 | 运维 | P1 | 1-2周 |

**最关键的技术挑战（非业务）：**
1. **GPU 推理延迟**：GRU 推理（128 hidden）在 2080 上约 10-50μs，但 Python→CUDA→GRU→sampling→CUDA→Python 的总 round-trip 可能 100-500μs
2. **JAX JIT 首次编译**：实时系统中不能等待 JIT 编译，需要 AOT 编译或预热
3. **模型热更新**：训练模型→部署模型的 pipeline 需要无缝

---

## 总结与优先级矩阵

### P0 (立即行动，阻塞当前进展)

| # | 建议 | 类别 | 预期收益 | 工作量 |
|---|------|------|---------|--------|
| 1 | 扩展 EXE reward 加入 VWAP 偏离惩罚 | Reward | EXE reward 改善 20-40% | 半天 |
| 2 | MM reward: gamma=0.1→0.3, 加入印花税 | Reward | 库存管理策略改善 | 1天 |
| 3 | 扩展观测至 14-16 维（+imbalance, spread_ratio, trend） | Obs | 策略质量本质提升 | 1-2天 |
| 4 | [10,10] agents: ENT_COEF 退火策略 | HP | 防止策略 collapse | 半天 |
| 5 | [10,10] agents: LR 降低到 1.5e-4 | HP | 训练稳定性 | 半天 |

### P1 (重要，下一阶段)

| # | 建议 | 类别 | 预期收益 | 工作量 |
|---|------|------|---------|--------|
| 1 | 代码拆分 mm_env.py → 5-6 个模块 | Code | 维护效率 2x | 2-3天 |
| 2 | 统一 pmap/非pmap 训练脚本 | Code | 减少代码重复 | 1-2天 |
| 3 | 独立 agent done bootstrap（agent-level GAE） | Algo | credit assignment 改善 | 1天 |
| 4 | 多天数据训练 | Data | 泛化能力 | 1周 |
| 5 | MM reward: inventory PnL gamma 分阶段 | Reward | 更好的策略学习 | 1天 |
| 6 | RunningMeanStd observation normalization | Env | 观测稳定性 | 半天 |

### P2 (优化，长期提升)

| # | 建议 | 类别 | 预期收益 | 工作量 |
|---|------|------|---------|--------|
| 1 | LayerNorm before GRU | Network | 训练稳定性 | 0.5天 |
| 2 | 多股票迁移学习 (Phase 2-3) | 泛化 | 实用性 | 2-4周 |
| 3 | JIT 编译优化（固定消息量上限） | Perf | 避免 re-JIT | 1天 |
| 4 | 创建 metrics.py 统一 logging | Code | 代码整洁 | 0.5天 |
| 5 | LSTM 替代 GRU 实验 | Network | 0-5% reward | 2天 |
| 6 | 订单簿多档深度纳入观测 | Obs | 策略丰富度 | 1-2天 |

### 工作量估算汇总

| 优先级 | 总工作量 |
|--------|---------|
| P0 (5项) | 约 3-5 天 |
| P1 (6项) | 约 7-10 天 |
| P2 (6项) | 约 1-6 周（视多股票范围） |

### 最核心的三个改进（ROI 最高）

1. **扩展观测空间** (P0#3)：用最小的代码改动获得最大的信息增益
2. **EXE reward 改进** (P0#1)：sleek 实验已证明 reward 信号改进能带来显著提升
3. **MM reward 参数调整** (P0#2)：T+1 reward 的 gamma 和成本参数需要 A 股实际校准

---

## G. 2026-05-22 新增：A 股视角盲区分析与修复建议

### G1. 背景

原代码库基于美股 AMZN / 高流动性假设设计。以下盲区来自迁移到 A 股（601398/601728）后的实证发现与分析。

### G2. 三个最高优先级盲区

#### G2.1 10MM+10EXE agent 数量严重过多（高优先级，只改配置）

**问题：** 低流动性 A 股（601728 中国电信）每步 60 条数据消息 + 80 条 agent 消息，20 个 RL agent 的订单占满最优盘口。训练结果验证：601728 EXE reward≈-0.184（无学习），601398 EXE≈-0.081（有学习）。

**修复：** 减到 2-3 MM + 2-3 EXE，或大幅降低 fixed_quant_value。

#### G2.2 T+1 未在动作层面真正约束（高优先级，需重跑）

**问题：** base_inventory/intraday_buys 在 state 中追踪，但动作函数（_getActionMsgs_*）中不检查这些约束。Agent 可以卖超出 base_inventory 的量，也可以卖出当日购入股。

**修复：** 下单前限制 sell_quant ≤ max(0, base_inventory)，买之前检查 cash_balance。

#### G2.3 EXE value_loss 爆炸的结构性原因（高优先级，需重跑）

**问题：** 10 个 EXE 在浅薄盘口上互相竞争，互相推高 VWAP 基准。VWAP 包含其他 RL agent 的成交。EXE value_loss≈18（601398）到≈27（601728）。

**修复：** 减少 EXE agent 数量，或将 VWAP 计算改为纯真实市场成交（排除其他 RL agent）。

### G3. 中优先级盲区

#### G3.1 Episode 结束未清算 base_inventory（中优先级，只改代码）

虚构交易只清算 net inventory，忽略 base_inventory（底仓 500 股）和 intraday_buys 的估值。

#### G3.2 虚假申报隐患 / OrderManager 覆盖不全（中优先级，只改代码）

OrderManager 已实现，但价格比较未使用 `price // tick_size` 可能产生边界误匹配。训练初期 agent 不断探索，有效撤单率仍~100%。

#### G3.3 低流动性下 doom_price_penalty 失效（中优先级，需重跑）

doom=1 在 A 股低流动性下几乎无影响（600 股 × ¥0.01 = ¥6 惩罚）。建议使用动态计算（取 spread 的一定比例）。

#### G3.4 GRU_HIDDEN_DIM=128 在低频环境下过参数化（中优先级，需重跑）

A 股低流动性数据稀疏，GRU-128 可能过拟合噪声。建议小批量尝试 GRU-64 + FC-256。

### G4. 低优先级 / 已确认正确的项

| 项 | 状态 | 备注 |
|:---|:----|:-----|
| stamp_duty 计算方向 | ✅ 正确 | 只收卖出，income×bps/10000 |
| commission 计算方向 | ✅ 正确 | 买卖双边，income+outgoing |
| rebate 代码路径 | ✅ 无害 | rebate_bps=0，但被动/主动区分仍每步计算 |
| tick_size=100 | ✅ 正确 | 对应 A 股 ¥0.01 最小变动 |
| cash_balance reset | ✅ 已修 | 初始化为 base_inventory × mid_price/tick_size |
| OmegaConf merge bug | ✅ 已确认 | AGENT_CONFIGS 被 dict_of_agent_configs 覆盖 |
| REWARD_NORM 实现错误 | ✅ 已确认 | 在 GAE 内部做 per-timestep z-score |

### G5. 快速可修项（不改训练结果）

1. Rebate 路径条件计算（rebate_bps>0 时才区分被动/主动）
2. YAML 中移除手动 n_actions / num_messages 覆盖（让 __post_init__ 自动计算）
3. OrderManager 价格比较使用 `price // tick_size`（确保整数匹配）

---

> 报告完成。以上分析基于对全部生产代码的逐行审查。
