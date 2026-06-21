"""
Unit tests for MarketMakingAgent.get_reward with reward_function="tplus1".

Tests cover:
  1. Empty position, no trades → reward = 0
  2. Buy-only (inventory change only, no sell)
  3. Buy + sell matched pair (positive spread capture)
  4. Inventory loss (asymmetric penalty)
  5. Overnight position penalty
  6. Forced unwind at episode end
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


# ---------------------------------------------------------------------------
# Helper: create a minimal agent + call get_reward
# ---------------------------------------------------------------------------

def reward_for(
    reward_function="tplus1",
    inventory=0,
    cash_balance=0.0,
    base_inventory=0,
    intraday_buys=0,
    bid_price=99_000,
    ask_price=100_000,
    trades_fn=None,       # callable(trader_id) → trades array
    ep_done_time=False,
    tick_size=100,
    commission_bps=1.0,
    stamp_duty_bps=5.0,
    rebate_bps=0.0,
    inventoryPnL_eta=0.5,
    inventoryPnL_gamma=0.3,
    overnight_penalty_lambda=0.0,
    initial_base_inventory=0,
    **cfg_kwargs,
):
    """Construct everything needed and call get_reward, return the scalar reward."""

    mm_cfg = make_config(
        reward_function=reward_function,
        commission_bps=commission_bps,
        stamp_duty_bps=stamp_duty_bps,
        rebate_bps=rebate_bps,
        inventoryPnL_eta=inventoryPnL_eta,
        inventoryPnL_gamma=inventoryPnL_gamma,
        overnight_penalty_lambda=overnight_penalty_lambda,
        initial_base_inventory=initial_base_inventory,
        **cfg_kwargs,
    )
    world_cfg = make_world_config(tick_size=tick_size)
    agent = MarketMakingAgent(mm_cfg, world_cfg)

    # Agent params
    trader_id = -1
    agent_params = make_agent_params(trader_id=trader_id)

    # World state & best prices
    best_bids, best_asks = make_best_prices(
        bid_price=bid_price, ask_price=ask_price
    )
    world_state = make_world_state(
        best_bids=best_bids,
        best_asks=best_asks,
        mid_price=(bid_price + ask_price) / 2,
        step_counter=10,
    )

    # Agent state
    agent_state = make_agent_state(
        inventory=inventory,
        cash_balance=cash_balance,
        base_inventory=base_inventory,
        intraday_buys=intraday_buys,
    )

    # Trades
    trades = trades_fn(trader_id) if trades_fn else make_empty_trades()

    reward, info = agent.get_reward(
        world_state=world_state,
        agent_state=agent_state,
        agent_params=agent_params,
        trades=trades,
        bestasks=best_asks,
        bestbids=best_bids,
        ep_done_time=ep_done_time,
    )

    # reward may be shape () or (1,) — extract scalar
    return float(jnp.squeeze(reward))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTplus1NoActivity:
    """Case 1: empty position, no trades → reward = 0 (or very near 0)."""

    def test_no_trades_no_inventory(self):
        r = reward_for(inventory=0, cash_balance=0.0)
        # commission/stamp are 0 because no trades happen
        assert abs(r) < 1e-6, f"Expected ~0, got {r}"

    def test_no_trades_with_inventory(self):
        """Holding inventory but no trades → only inventory PnL (which is 0 if mid unchanged)."""
        r = reward_for(inventory=10, cash_balance=0.0)
        # No trades → buyPnL=sellPnL=0, inventory unchanged → InventoryPnL ≈ 0
        assert abs(r) < 1e-4, f"Expected ~0, got {r}"


class TestTplus1WithTrades:
    """Cases involving actual trades."""

    def test_buy_only_adds_inventory(self):
        """
        Agent buys (passive) some quantity, mid price unchanged.
        Reward should be small (mid price hasn't moved → no PnL from inventory).
        """
        def make_trades(tid):
            # Agent passively buys 10 shares at 99_500
            return make_single_trade(price=99_500, quantity=10, pass_tid=tid)

        r = reward_for(inventory=0, cash_balance=0.0, trades_fn=make_trades)
        # Passive buy at mid price → no profit/loss from price, small commission cost
        # commission_bps=1.0 on 10*99_500/100 ticks * 1 tick=?
        # Actually buy cost in ticks: 99_500/100 = 995 ticks
        # income = 995 * 10 = 9950 (outgoing since it's a buy)
        # commission = 9950 * (1/10000) = 0.995
        # With no sell, buyPnL=0, sellPnL=0, InventoryPnL depends on mid price
        # Since mid stayed same, InventoryPnL=0
        # reward ≈ -commission ≈ -1.0
        assert r < 0 and r > -5, f"Expected small negative (commission cost), got {r}"

    def test_buy_sell_spread_capture(self):
        """
        Agent buys at bid price and sells at ask price → captures spread.
        Expected positive reward.
        """
        def make_trades(tid):
            # Buy 10 at bid (passive) → earns rebate? No rebate_bps=0
            # Sell 10 at ask (passive)
            trades = make_single_trade(price=99_000, quantity=10, pass_tid=tid)
            trades = trades.at[1, 0].set(100_000)  # sell price
            trades = trades.at[1, 1].set(-10)       # sell qty (negative = sell)
            trades = trades.at[1, 6].set(tid)        # passive TID
            trades = trades.at[1, 7].set(-2)         # aggr TID
            return trades

        r = reward_for(inventory=0, cash_balance=0.0, trades_fn=make_trades)
        # Spread captured: sell at 100_000 - buy at 99_000 = 1000 profit in raw price
        # In ticks: 1000/100 = 10 ticks * 10 shares = 100 tick-profit
        # Minus commission ≈ 2 * ~1.0 = -2
        # Expected positive (~98ish)
        assert r > 50, f"Expected positive (spread capture), got {r}"


class TestTplus1Penalties:
    """Penalty mechanisms."""

    def test_commission_reduces_reward(self):
        """
        Higher commission_bps → lower reward (direct cost subtracted).
        """
        def make_trades(tid):
            # Buy at bid, sell at ask — generates taxable income
            trades = make_single_trade(price=99_000, quantity=10, pass_tid=tid)
            trades = trades.at[1, 0].set(100_000)
            trades = trades.at[1, 1].set(-10)
            trades = trades.at[1, 6].set(tid)
            trades = trades.at[1, 7].set(-2)
            return trades

        r_low_comm = reward_for(
            commission_bps=1.0,
            trades_fn=make_trades,
        )
        r_high_comm = reward_for(
            commission_bps=10.0,
            trades_fn=make_trades,
        )
        assert r_high_comm < r_low_comm, (
            f"Higher commission should reduce reward: "
            f"1bps -> {r_low_comm}, 10bps -> {r_high_comm}"
        )

    def test_overnight_penalty(self):
        """
        When overnight_penalty_lambda > 0 and there's locked intraday position,
        reward should be reduced.
        """
        def make_trades(tid):
            return make_empty_trades()

        r_no_penalty = reward_for(
            inventory=0, cash_balance=0.0,
            base_inventory=0, intraday_buys=0,
            overnight_penalty_lambda=0.0,
            trades_fn=make_trades,
        )
        r_with_penalty = reward_for(
            inventory=0, cash_balance=0.0,
            base_inventory=0, intraday_buys=100,  # locked intraday position
            overnight_penalty_lambda=0.01,
            trades_fn=make_trades,
        )
        # No trades → no pnl changes, overnight penalty subtracts
        assert r_with_penalty < r_no_penalty, (
            f"Overnight penalty should reduce reward: "
            f"no_penalty={r_no_penalty}, with_penalty={r_with_penalty}"
        )


class TestTplus1Unwind:
    """Forced unwind at episode end."""

    def test_forced_unwind_reduces_reward(self):
        """
        When ep_done_time=True and inventory != 0,
        a fictional trade at disadvantageous price is added,
        reducing the reward.
        """
        def make_trades(tid):
            return make_empty_trades()

        # Holding 10 long → forced to sell at bid price with penalty
        r_no_unwind = reward_for(
            inventory=10, cash_balance=0.0,
            ep_done_time=False,
            trades_fn=make_trades,
        )
        r_unwind = reward_for(
            inventory=10, cash_balance=0.0,
            ep_done_time=True,
            trades_fn=make_trades,
        )
        assert r_unwind <= r_no_unwind, (
            f"Forced unwind should reduce reward: "
            f"no_unwind={r_no_unwind}, unwind={r_unwind}"
        )
