"""
Unit tests for MarketMakingAgent.get_reward with reward_function="spooner_asym_damped2".

Tests cover:
  1. No activity → reward = 0
  2. Buy + sell spread capture (asymmetric damping shouldn't affect positive PnL much)
  3. Inventory loss damping (asymmetric: positive InventoryPnL is damped, negative is not)
"""

import jax.numpy as jnp
import pytest

from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from tests.conftest import (
    make_config,
    make_world_config,
    make_world_state,
    make_agent_state,
    make_agent_params,
    make_best_prices,
    make_single_trade,
    make_empty_trades,
)


def reward_for(
    reward_function="spooner_asym_damped2",
    inventory=0,
    cash_balance=0.0,
    bid_price=99_000,
    ask_price=100_000,
    old_mid_price=None,        # ← allow overriding old mid price
    trades_fn=None,
    ep_done_time=False,
    tick_size=100,
    inventoryPnL_eta=0.8,
    inventoryPnL_gamma=0.3,
    **cfg_kwargs,
):
    """Helper — same pattern as tplus1 tests."""
    mm_cfg = make_config(
        reward_function=reward_function,
        inventoryPnL_eta=inventoryPnL_eta,
        inventoryPnL_gamma=inventoryPnL_gamma,
        **cfg_kwargs,
    )
    world_cfg = make_world_config(tick_size=tick_size)
    agent = MarketMakingAgent(mm_cfg, world_cfg)

    trader_id = -1
    agent_params = make_agent_params(trader_id=trader_id)
    best_bids, best_asks = make_best_prices(bid_price=bid_price, ask_price=ask_price)
    if old_mid_price is None:
        old_mid_price = (bid_price + ask_price) / 2
    world_state = make_world_state(
        best_bids=best_bids, best_asks=best_asks,
        mid_price=old_mid_price,
    )
    agent_state = make_agent_state(inventory=inventory, cash_balance=cash_balance)
    trades = trades_fn(trader_id) if trades_fn else make_empty_trades()

    reward, info = agent.get_reward(
        world_state=world_state, agent_state=agent_state,
        agent_params=agent_params, trades=trades,
        bestasks=best_asks, bestbids=best_bids,
        ep_done_time=ep_done_time,
    )
    return float(jnp.squeeze(reward))


class TestSpoonerAsymDamped2:

    def test_no_activity(self):
        """No trades, no inventory → reward ≈ 0."""
        r = reward_for(inventory=0)
        assert abs(r) < 1e-6, f"Expected ~0, got {r}"

    def test_spread_capture(self):
        """Buy low, sell high → positive reward."""
        def make_trades(tid):
            trades = make_single_trade(price=99_000, quantity=10, pass_tid=tid)
            trades = trades.at[1, 0].set(100_000)
            trades = trades.at[1, 1].set(-10)
            trades = trades.at[1, 6].set(tid)
            trades = trades.at[1, 7].set(-2)
            return trades

        r = reward_for(inventory=0, trades_fn=make_trades)
        assert r > 50, f"Spread capture should be clearly positive, got {r}"

    def test_asymmetric_damping(self):
        """
        spooner_asym_damped2 = buyPnL + sellPnL + rebate
            + gamma*(InventoryPnL - max(0, eta*InventoryPnL))

        With positive inventory and no mid-price change, InventoryPnL is positive.
        Higher inventoryPnL_eta → more positive portion damped → lower reward.

        With negative inventory, InventoryPnL is negative → jnp.maximum(0, eta*neg)=0
        → no damping regardless of eta.
        """
        def make_trades_long(tid):
            return make_single_trade(price=99_500, quantity=1, pass_tid=tid)

        def make_trades_short(tid):
            return make_single_trade(price=99_500, quantity=-1, pass_tid=tid)

        # Long position: make mid price rise → positive InventoryPnL → damping applies
        r_eta_low = reward_for(
            inventory=100, cash_balance=0.0,
            old_mid_price=99_000,          # old mid was 99_000
            inventoryPnL_eta=0.2,           # new mid from best_prices is 99_500
            trades_fn=make_trades_long,
        )
        r_eta_high = reward_for(
            inventory=100, cash_balance=0.0,
            old_mid_price=99_000,
            inventoryPnL_eta=0.9,
            trades_fn=make_trades_long,
        )
        # Higher eta → more damping → lower reward
        assert r_eta_high < r_eta_low, (
            f"Long: higher eta should damp more: "
            f"eta=0.2 -> {r_eta_low}, eta=0.9 -> {r_eta_high}"
        )

        # Short position: InventoryPnL < 0 → no damping (max(0, eta*neg) ≡ 0)
        r_short_eta_low = reward_for(
            inventory=-100, cash_balance=0.0,
            inventoryPnL_eta=0.2,
            trades_fn=make_trades_short,
        )
        r_short_eta_high = reward_for(
            inventory=-100, cash_balance=0.0,
            inventoryPnL_eta=0.9,
            trades_fn=make_trades_short,
        )
        # Short: formula becomes gamma*(InventoryPnL - 0) → eta irrelevant
        assert abs(r_short_eta_high - r_short_eta_low) < 1e-4, (
            f"Short: eta should not matter when InventoryPnL < 0: "
            f"eta=0.2 -> {r_short_eta_low}, eta=0.9 -> {r_short_eta_high}"
        )
