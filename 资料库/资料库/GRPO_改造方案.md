# GRPO 改造方案 — JaxMARL-HFT

## 概述

将当前 IPPO+PPO+LSTM 训练架构改造为 GRPO（Group Relative Policy Optimization），
基于 DeepSeek-R1 的核心思想：**去掉价值网络（critic），用组内 reward 归一化替代 GAE**。

## 核心改动

### 1. 网络架构：ActorCriticRNN → ActorOnlyRNN

| | PPO | GRPO |
|---|---|---|
| 输出 | `(hidden, pi, value)` | `(hidden, pi)` |
| critic 头 | `nn.Dense(FC_DIM) → nn.Dense(1)` | 删除 |
| 参数量 | 2 个输出头 | 1 个输出头 |

### 2. Advantage 计算：GAE → Group Normalization

**PPO (GAE)**:
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ (γλ)^(l-t)·δ_l
```
需要价值网络、bootstrap、超参 γ 和 λ。

**GRPO (Group Advantage)**:
```
A_i = (r_i - μ_group) / σ_group
```
无需价值网络，无需 γ/λ，天然适配多 agent 场景。

### 3. Loss 函数：去掉价值损失，加 KL 惩罚

**PPO loss**:
```
L = -L_clip + c_v·L_value - c_e·H(π)
```

**GRPO loss**:
```
L = -L_clip + β·KL(π_ref || π) - c_e·H(π)
```

KL 惩罚使用近似公式：`KL ≈ (π_ref/π) - log(π_ref/π) - 1`

### 4. Reference Policy

在每次 minibatch 更新开始时，快照当前 policy 参数作为 π_ref，
在 loss 中计算 KL 惩罚防止策略漂移。

## 改动文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `ippo_rnn_JAXMARL_GRPO_pmap.py` | ✅ 已创建 | 基于 LSTM PMAP 的 GRPO 版本 |
| `config/.../XXX.yaml` | 需创建 | 新配置，用 KL_COEF 替换 VF_COEF |

## YAML 配置变化

**移除**:
- `VF_COEF`（不再需要价值网络）
- `GAMMA`、`GAE_LAMBDA`（不再需要 GAE）

**新增**:
- `KL_COEF`：KL 惩罚系数（对应 β），建议起始值 `[0.01, 0.01]`
- `TRAIN_ALGO: "grpo"`（可选，用于区分）

**保留**:
- `ENT_COEF`、`CLIP_EPS`、`LR` 等 PPO 原有超参不变

示例：
```yaml
# 原来的
"VF_COEF": [0.1, 0.01]
"GAMMA": [0.99999, 0.99]
"GAE_LAMBDA": [0.85, 0.95]

# 改为
"KL_COEF": [0.01, 0.01]
```

## 运行命令

```bash
cd ~/JaxMARL-HFT

python gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL_GRPO_pmap.py \
  --config-path=../../../config/rl_configs/300059 \
  --config-name=agents10_5act \
  'VF_COEF=null' 'GAE_LAMBDA=null' 'GAMMA=null' 'KL_COEF=[0.01,0.01]'
```

## 预期效果

1. **EXE value_loss 爆炸问题消失** — 因为没有 critic
2. **10 EXE agent 之间竞争明确** — 组内归一化让 agent 直接相互比较
3. **训练更稳定** — 少一个 critic 的参数量 + 少两个超参
4. **MM side 同理** — 10 MM agent 也是一个 group

## 注意事项

1. **Group size 不能太小** — 当前 [10,10] 足够，减到 [2,2] 可能组内归一化不稳定
2. **参考策略的 KL** — β 太大策略学不动，太小策略漂移。建议先从 0.01 试起
3. **无需单独训 GRPO 版 baseline** — 直接用训练中的 π 自参考
