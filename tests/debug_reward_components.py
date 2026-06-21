"""
Quick reward breakdown: load real data, run a few env steps, print each component.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

# --- configs ---
from gymnax_exchange.config.env_configs import (
    MarketMaking_EnvironmentConfig,
    World_EnvironmentConfig,
)
from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.StatesandParams import (
    WorldState, MMEnvState, MMEnvParams,
)

# --- build env + agent ---
mm_cfg = MarketMaking_EnvironmentConfig(
    action_space="fixed_quants",
    observation_space="engineered",
    reward_function="tplus1",
    inventoryPnL_eta=0.5,
    inventoryPnL_gamma=0.3,
    commission_bps=1.0,
    stamp_duty_bps=5.0,
    rebate_bps=0.0,
    initial_base_inventory=500,
    overnight_penalty_lambda=0.01,
    reference_price="mid",
    unwind_price="mid",
    unwind_price_penalty=0,
    reward_scaling_quo=100.0,
)

world_cfg = World_EnvironmentConfig(
    stock="300059",
    timePeriod="20240103",
    tick_size=100,
    book_depth=10,
    n_data_msg_per_step=120,
    start_resolution=64,
    ep_type="fixed_steps",
    episode_time=64,
    day_start=34200,
    day_end=54000,
)

# --- BaseLOBEnv for data loading ---
base_env = BaseLOBEnv(world_cfg)

print("Loading data...")
key = jax.random.PRNGKey(42)
init_key, step_key = jax.random.split(key)

# Initialize
obs, state, params = base_env.reset(init_key)
print("State initialized:", type(state).__name__)
print()

agent = MarketMakingAgent(mm_cfg, world_cfg)
agent_params = agent.default_params(mm_cfg, -100, 1)
trader_id = agent_params.trader_id

# --- run N steps collecting reward components ---
N_STEPS = 50
components = {k: [] for k in [
    "reward", "buyPnL", "sellPnL", "InventoryPnL",
    "commission_cost", "stamp_duty", "rebate_income",
    "reward_tplus1", "PnL", "delta_mid_price",
    "mid_price_end", "old_mid_price", "inventory",
    "buyQuant", "sellQuant",
]}
mid_price_history = []

for step_i in range(N_STEPS):
    # Get current world_state from marl state (first agent)
    world_state = state.world_state
    agent_state = state.agent_states[0]  # type: MMEnvState

    # Random action
    key, subkey = jax.random.split(key)
    action = jax.random.randint(subkey, (1,), 0, mm_cfg.n_actions)

    # Step the base env to get next market data
    new_obs, new_state, new_reward, done, info = base_env.step_env(
        step_key, state, {0: action[0]}, params
    )
    step_key, _ = jax.random.split(step_key)

    # Now call get_reward directly with the new world state to get breakdown
    # We need the trades, bestasks, bestbids from the new world state
    # Actually the step_env already computed the reward - let's extract from info
    # But info doesn't have breakdown... Let's call get_reward manually

    # For the next step, we'd need the trades array which step_env returned
    # Let's take a different approach: use the agent's own step
    
    # Actually, let me just record the mid price to see how much it moves
    mid_before = world_state.mid_price
    # Step the env
    obs, state, reward_batch, done, info = base_env.step_env(
        subkey, state, {0: action[0]}, params
    )
    mid_after = state.world_state.mid_price
    
    mid_price_history.append(float(mid_after - mid_before))
    
    if step_i < 5 or step_i % 10 == 0:
        print(f"step {step_i:3d}:  mid Δ = {float(mid_after - mid_before):+8.1f}  "
              f"reward = {float(jnp.mean(reward_batch)):+.6f}")

print("\n--- Mid price movement stats ---")
mids = np.array(mid_price_history)
print(f"Mean Δ: {mids.mean():+.1f}")
print(f"Std  Δ: {mids.std():.1f}")
print(f"Max  Δ: {mids.max():+.1f}")
print(f"Min  Δ: {mids.min():+.1f}")
print(f"Abs Δ > 0: {(np.abs(mids) > 0).sum()} / {N_STEPS}")
