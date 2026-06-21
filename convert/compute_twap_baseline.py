"""
TWAP 基线计算脚本。
使用 5-act 动作空间，固定 action=1 (NT/被动价格)，
每步均匀分配剩余任务量。
"""
import os, sys, time, jax, jax.numpy as jnp, numpy as np
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

# JaxMARL-HFT 路径
sys.path.insert(0, '/home/atomus/JaxMARL-HFT')
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import (
    MultiAgentConfig, MarketMaking_EnvironmentConfig,
    Execution_EnvironmentConfig, World_EnvironmentConfig,
)

# === 配置（与 Phase 2 一致）===
stock = '300059'
ma_config = MultiAgentConfig(
    number_of_agents_per_type=[10, 1],
    dict_of_agents_configs={
        "MarketMaking": MarketMaking_EnvironmentConfig(
            action_space="fixed_quants", observation_space="engineered",
            reward_function="tplus1", inventoryPnL_eta=0.5,
            fixed_quant_value=6, initial_base_inventory=500,
            commission_bps=1.0, stamp_duty_bps=5,
            use_order_manager=True,
        ),
        "Execution": Execution_EnvironmentConfig(
            action_space="fixed_quants_5act",
            observation_space="engineered", reward_function="normal",
            task_size=1000, fixed_quant_value=30,
            reward_scaling_quo=100.0,
        ),
    },
    world_config=World_EnvironmentConfig(
        stock=stock, timePeriod='20240103',
        day_start=34200, day_end=54000,
        tick_size=100, book_depth=10,
        n_data_msg_per_step=120,
        start_resolution=64, ep_type="fixed_steps", episode_time=64,
    ),
)

key = jax.random.PRNGKey(42)
env = MARLEnv(key=key, multi_agent_config=ma_config)
rng = jax.random.PRNGKey(0)

# 初始化环境
reset_rng = jax.random.split(rng, 128)
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env.default_params)

# 一步环境交互：EXE 固定 action=1（NT/被动），MM 随机
def run_step(carry, unused):
    env_state, rng = carry
    rng, rng_act = jax.random.split(rng)

    mm_action = jax.random.randint(rng_act, (128, 10), 0, 10)
    exe_action = jnp.ones((128, 1), dtype=jnp.int32)  # 固定 action=1 (NT)

    rng_step = jax.random.split(rng, 128)
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, env_state, [mm_action, exe_action], env.default_params)

    return (env_state, rng), reward

rng = jax.random.PRNGKey(1)
(env_state, _), rewards = jax.lax.scan(run_step, (env_state, rng), None, 64)
mm_reward = jnp.mean(rewards[0])
exe_reward = jnp.mean(rewards[1])

print(f'=== TWAP 基线结果 (5-act, NT定价, 64步) ===')
print(f'MM reward:  {mm_reward:.6f}')
print(f'EXE reward: {exe_reward:.6f}')
print(f'(对比 Phase 2 学到的 EXE: -0.105)')
