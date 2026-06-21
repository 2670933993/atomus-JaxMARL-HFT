"""
Shared test fixtures for MM reward unit tests.

Constructs minimal synthetic JAX arrays so get_reward can be tested
without loading real LOBSTER data.
"""

import jax.numpy as jnp
from flax import struct

from gymnax_exchange.jaxen.StatesandParams import (
    WorldState, MMEnvState, MMEnvParams,
)
from gymnax_exchange.config.env_configs import (
    MarketMaking_EnvironmentConfig,
    World_EnvironmentConfig,
    CONFIG_OBJECT_DICT,
)


def make_config(
    reward_function: str = "tplus1",
    inventoryPnL_eta: float = 0.5,
    inventoryPnL_gamma: float = 0.3,
    tick_size: int = 100,
    rebate_bps: float = 0.0,
    commission_bps: float = 1.0,
    stamp_duty_bps: float = 5.0,
    initial_base_inventory: int = 0,
    overnight_penalty_lambda: float = 0.0,
    reference_price: str = "mid",
    unwind_price: str = "mid",
    unwind_price_penalty: int = 5,
    **kwargs,
) -> MarketMaking_EnvironmentConfig:
    """Create a MarketMaking_EnvironmentConfig with test-friendly defaults."""
    return MarketMaking_EnvironmentConfig(
        action_space="fixed_quants",
        observation_space="engineered",
        reward_function=reward_function,
        inventoryPnL_eta=inventoryPnL_eta,
        inventoryPnL_gamma=inventoryPnL_gamma,
        rebate_bps=rebate_bps,
        commission_bps=commission_bps,
        stamp_duty_bps=stamp_duty_bps,
        initial_base_inventory=initial_base_inventory,
        overnight_penalty_lambda=overnight_penalty_lambda,
        reference_price=reference_price,
        unwind_price=unwind_price,
        unwind_price_penalty=unwind_price_penalty,
        **kwargs,
    )


def make_world_config(
    tick_size: int = 100,
    book_depth: int = 10,
) -> World_EnvironmentConfig:
    """Create a World_EnvironmentConfig with test-friendly defaults."""
    return World_EnvironmentConfig(
        tick_size=tick_size,
        book_depth=book_depth,
        stock="TEST",
        timePeriod="TEST",
    )


def make_best_prices(
    bid_price: float = 99_000,
    ask_price: float = 100_000,
    num_levels: int = 10,
) -> tuple:
    """
    Create best_bids and best_asks arrays.
    Returns (best_bids, best_asks) each with shape (num_levels, 2).
    Column 0 = price, Column 1 = quantity.
    """
    best_bids = jnp.zeros((num_levels, 2), dtype=jnp.int32)
    best_asks = jnp.zeros((num_levels, 2), dtype=jnp.int32)

    # Fill levels with descending bid prices / ascending ask prices
    for i in range(num_levels):
        best_bids = best_bids.at[i, 0].set(bid_price - i * 100)
        best_bids = best_bids.at[i, 1].set(100)  # constant qty
        best_asks = best_asks.at[i, 0].set(ask_price + i * 100)
        best_asks = best_asks.at[i, 1].set(100)

    return best_bids, best_asks


def make_world_state(
    best_bids: jnp.ndarray,
    best_asks: jnp.ndarray,
    mid_price: float = 99_500,
    step_counter: int = 10,
    time_sec: int = 34_500,
    delta_time: float = 1.0,
) -> WorldState:
    """Create a WorldState with synthetic data."""
    return WorldState(
        # LoadedEnvState required fields (minimal dummies)
        ask_raw_orders=jnp.zeros((1, 1, 1, 6), dtype=jnp.int32),
        bid_raw_orders=jnp.zeros((1, 1, 1, 6), dtype=jnp.int32),
        trades=jnp.zeros((1, 1, 1, 8), dtype=jnp.int32),
        init_time=jnp.array([time_sec, 0], dtype=jnp.int32),
        window_index=0,
        max_steps_in_episode=100,
        start_index=0,
        step_counter=step_counter,
        # WorldState specific
        best_bids=best_bids,
        best_asks=best_asks,
        time=jnp.array([time_sec, 0], dtype=jnp.int32),
        order_id_counter=0,
        mid_price=mid_price,
        delta_time=delta_time,
    )


def make_agent_state(
    inventory: int = 0,
    total_PnL: float = 0.0,
    cash_balance: float = 0.0,
    base_inventory: int = 0,
    intraday_buys: int = 0,
    base_cost_basis: float = 0.0,
    posted_distance_bid: int = 0,
    posted_distance_ask: int = 0,
) -> MMEnvState:
    """Create an MMEnvState with synthetic data."""
    return MMEnvState(
        posted_distance_bid=posted_distance_bid,
        posted_distance_ask=posted_distance_ask,
        inventory=inventory,
        total_PnL=total_PnL,
        cash_balance=cash_balance,
        base_inventory=base_inventory,
        intraday_buys=intraday_buys,
        base_cost_basis=base_cost_basis,
    )


def make_agent_params(
    trader_id: int = -1,
    normalize: bool = True,
    time_delay_obs_act: int = 0,
) -> MMEnvParams:
    """Create MMEnvParams — trader_id as scalar (test uses direct call, not vmap)."""
    return MMEnvParams(
        trader_id=jnp.array(trader_id),
        normalize=jnp.array([normalize]),
        time_delay_obs_act=jnp.array([time_delay_obs_act]),
    )


def make_empty_trades(n_max: int = 10) -> jnp.ndarray:
    """
    Create a zero-filled trade array of shape (n_max, 8).
    Columns follow TradesFeat order: Price, Q, PassOID, AgrOID, Sec, NSec, PassTID, AgrTID.
    """
    return jnp.zeros((n_max, 8), dtype=jnp.int32)


def make_single_trade(
    price: int,
    quantity: int,
    pass_tid: int = -1,
    aggr_tid: int = -2,
    n_max: int = 10,
) -> jnp.ndarray:
    """
    Create a trade array with one trade at index 0, rest zero.
    quantity > 0 = buy for passive, quantity < 0 = sell for passive.
    """
    trades = jnp.zeros((n_max, 8), dtype=jnp.int32)
    trades = trades.at[0, 0].set(price)      # Price
    trades = trades.at[0, 1].set(quantity)   # Qty
    trades = trades.at[0, 6].set(pass_tid)    # Passive TID
    trades = trades.at[0, 7].set(aggr_tid)     # Aggressive TID
    return trades
