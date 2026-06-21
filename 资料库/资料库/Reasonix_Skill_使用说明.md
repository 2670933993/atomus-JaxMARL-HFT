# Reasonix Skills 使用说明

## 什么是 Skill

Skill 是 Reasonix 的可复用 playbook。每个 Skill 定义了一个特定任务的工作流，调用后自动执行。有些 Skill 是**内联执行**（输出直接出现在当前对话），有些是**子 agent 执行**（隔离运行，只返回结论）。

---

## 技能列表

| 技能名 | 调用方式 | 功能 | 类型 |
|--------|---------|------|------|
| `explore` | `/skill explore <任务>` | 只读代码调查，跨文件搜索调用关系 | 子 agent |
| `research` | `/skill research <问题>` | 结合代码阅读 + 网络搜索调研 | 子 agent |
| `review` | `/skill review <范围>` | 审查当前分支变更的正确性/安全性 | 子 agent |
| `security-review` | `/skill security-review` | 安全专项审查（注入/认证/密钥等） | 子 agent |
| `test` | `/skill test` | 自动识别框架、跑测试、诊断失败 | 内联 |
| `refactor` | `/skill refactor <目标>` | 重构前影响分析，追踪调用链和依赖风险 | 子 agent |
| `document` | `/skill document <路径>` | 批量给函数加注释或生成 README | 子 agent |

---

## 用法示例

### 1. 代码调查

```bash
# 探索 env_step 函数的调用链
/skill explore mm_env.py 里的 _get_reward 被哪些函数调用
```

### 2. 重构前分析

```bash
/skill refactor 把 marl_env.py 的 step 函数拆成三个小函数
```

输出示例：
```
受影响文件:
  - marl_env.py (定义)
  - ippo_rnn_JAXMARL_pmap.py (第 342 行调用)
  - mm_env.py (第 156 行被委派)
风险: 中 — 被 pmap 编译引用，需重编译
```

### 3. 批量加注释

```bash
/skill document gymnax_exchange/jaxen/mm_env.py
```

或针对模块：
```bash
/skill document gymnax_exchange/jaxob/
```

### 4. 审查代码

```bash
# 审查当前分支所有改动
/skill review

# 只审查某个模块
/skill review focus on jaxrl/
```

### 5. 调试测试

```bash
/skill test
```

---

## 最佳实践

| 场景 | 推荐 Skill | 为什么 |
|------|-----------|--------|
| "这段代码是干什么的" | `explore` | 只读，不污染上下文 |
| "想改这个函数，会不会出事" | `refactor` | 先分析调用链再动手 |
| "这个 PR 有没有问题" | `review` | 安全性 + 正确性双重检查 |
| "代码没有注释看不懂" | `document` | 批量加，不改逻辑 |
| "测试挂了不知道为什么" | `test` | 自动识别框架 + 修复后重跑 |

---

## 注意事项

- 子 agent Skill 的**工具调用不进入当前对话**，只看最终结论
- 如果需要跟踪中间结果，手动执行对应步骤而不是调 Skill
- Skill 会在下次 `/new` 后出现在技能索引中，当前会话可直接使用
