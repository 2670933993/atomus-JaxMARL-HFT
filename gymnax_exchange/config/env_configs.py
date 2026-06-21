"""
Environment configuration dataclasses.

Previously defined in gymnax_exchange/jaxob/jaxob_config.py.
All config schemas are consolidated here.
"""

import os
from typing import OrderedDict, Tuple, Literal, Union, List
from dataclasses import dataclass, field

import gymnax_exchange.jaxob.jaxob_constants as cst


# ──────────────────────────────────────────────
# Core Exchange Configuration
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class JAXLOB_Configuration:
    maxint: int = cst.MaxInt._64_Bit_Signed.value
    init_id: int = cst.INITID
    book_depth: int = 10
    cancel_mode: int = cst.CancelMode.INCLUDE_INITS.value
    type_4_interpretation: int = cst.Type4Interpretation.IOC.value
    seed: int = cst.SEED
    nTrades: int = cst.NTRADE_CAP
    nOrders: int = cst.NORDER_CAP
    simulator_mode: int = cst.SimulatorMode.GENERAL_EXCHANGE.value
    empty_slot_val: int = cst.EMPTY_SLOT
    debug_mode: bool = False
    check_book_fill: bool = True
    start_resolution: int = 6400
    alphatradePath: str = os.path.expanduser("~")
    dataPath: str = os.path.expanduser("~") + "/data"
    stock: str = "AMZN"
    timePeriod: str = "2024_Dec"


@dataclass(frozen=True)
class MarketMaking_EnvironmentConfig:
    # Debugging options (incl Simple Act Space)
    debug_mode: bool = False
    short_name: str = "MM"
    normalize: bool = True
    clip_reward: bool = False
    exclude_extreme_spreads: bool = False

    fixed_action_setting: bool = False
    fixed_action: int = 0
    simple_nothing_action: bool = True
    sell_buy_all_option: bool = False
    based_on_mid_price_of_action: bool = True
    tenth_action: str = "MarketOrder"
    bob_v0: int = 1

    # Real Parameters
    action_space: str = "bobRL"
    observation_space: str = "engineered"
    reward_function: str = "spooner_asym_damped2"

    # Values for action space
    spread_multiplier: float = 3.0
    skew_multiplier: float = 5.0
    n_ticks_offset: int = 1
    fixed_quant_value: int = 10
    auto_liquidate_threshold: int = 10000
    auto_liquidate_alpha: float = 1.0

    # Reward
    unwind_price_penalty: int = 5
    inv_penalty: str = "none"
    volume_traded_bonus: str = "none"
    reference_price: str = "mid"
    unwind_price: str = "mid"
    inv_penalty_lambda: float = 1.0
    inv_penalty_quadratic_factor: float = 50.0
    inv_penalty_threshold: float = 10.0
    multiplier_type: str = "tick"
    reward_scaling_quo: float = 1.0
    inventoryPnL_eta: float = 0.8
    inventoryPnL_gamma: float = 0.3

    rebate_bps: float = 10.0

    # T+1 settlement / A股 params
    initial_base_inventory: int = 0
    commission_bps: float = 0.0
    stamp_duty_bps: float = 10.0
    overnight_penalty_lambda: float = 0.0

    # Weights for complex reward function
    unrealizedPnL_lambda: float = 0.1
    avst_k_parameter: float = 0.4
    avst_var_parameter: float = 1e-8

    # Not actually implemented yet
    time_delay_obs_act: int = 0

    # OrderManager: compare target vs current orders; skip redundant cancel-repost
    use_order_manager: bool = False

    # Set Automatically in Post Init based on action space.
    n_actions: int = 10
    num_messages_by_agent: int = 4
    num_action_messages_by_agent: int = 2

    def __post_init__(self):
        if self.action_space == "fixed_quants":
            if self.tenth_action == "NA":
                object.__setattr__(self, 'n_actions', 9)
            elif self.tenth_action == "MarketOrder":
                object.__setattr__(self, 'n_actions', 10)
            else:
                raise ValueError(f"Invalid tenth_action {self.tenth_action} for fixed_quants action space")
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "spread_skew":
            object.__setattr__(self, 'n_actions', 6)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "bobStrategy":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "bobRL":
            if self.bob_v0 == 1:
                object.__setattr__(self, 'n_actions', 3)
            elif self.bob_v0 == 2:
                object.__setattr__(self, 'n_actions', 5)
            elif self.bob_v0 == 5:
                object.__setattr__(self, 'n_actions', 11)
            elif self.bob_v0 == 10:
                object.__setattr__(self, 'n_actions', 21)
            else:
                raise ValueError(f"Invalid bob_v0 {self.bob_v0} for bobRL action space")
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "directional_trading":
            object.__setattr__(self, 'n_actions', 3)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "AvSt":
            object.__setattr__(self, 'n_actions', 8)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "fixed_prices":
            object.__setattr__(self, 'num_messages_by_agent', self.n_actions * 2)
            object.__setattr__(self, 'num_action_messages_by_agent', self.n_actions)


@dataclass(frozen=True)
class Execution_EnvironmentConfig:
    debug_mode: bool = False
    larger_far_touch_quant: bool = False
    normalize: bool = True
    short_name: str = "EXE"
    action_type: str = "pure"

    # Real Parameters
    task: str = "random"
    action_space: str = "fixed_quants_complex"
    observation_space: str = "engineered"
    reward_function: str = "normal"
    task_size: int = 600
    n_ticks_in_book: int = 1
    fixed_quant_value: int = 10
    reward_lambda: float = 0.0
    reward_scaling_quo: float = 1.0
    doom_price_penalty: int = 5
    reference_price: str = "mid"

    # OrderManager: compare target vs current orders; skip redundant cancel-repost
    use_order_manager: bool = False

    # Not functional.. yet
    time_delay_obs_act: int = 0

    # Set Automatically in Post Init based on action space.
    n_actions: int = 5
    num_messages_by_agent: int = 8
    num_action_messages_by_agent: int = 4

    def __post_init__(self):
        if self.action_space == "fixed_quants":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 8)
            object.__setattr__(self, 'num_action_messages_by_agent', 4)
        elif self.action_space == "fixed_prices":
            object.__setattr__(self, 'num_messages_by_agent', self.n_actions * 2)
            object.__setattr__(self, 'num_action_messages_by_agent', self.n_actions)
        elif self.action_space == "fixed_quants_complex":
            object.__setattr__(self, 'n_actions', 13)
            object.__setattr__(self, 'num_messages_by_agent', 8)
            object.__setattr__(self, 'num_action_messages_by_agent', 4)
        elif self.action_space == "fixed_quants_5act":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "simplest_case":
            object.__setattr__(self, 'n_actions', 3)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "fixed_quants_1msg":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 2)
            object.__setattr__(self, 'num_action_messages_by_agent', 1)
        elif self.action_space == "twap":
            object.__setattr__(self, 'n_actions', 1)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)


@dataclass(frozen=True)
class World_EnvironmentConfig(JAXLOB_Configuration):
    n_data_msg_per_step: int = 1
    window_selector: int = -1
    ep_type: str = "fixed_steps"
    episode_time: int = 6400
    day_start: int = 34200
    day_end: int = 57600
    tick_size: int = 100
    trader_id_range_start: int = -100
    placeholder_order_id: int = -198
    artificial_trader_id_end_episode: int = -199
    artificial_order_id_end_episode: int = -199
    any_message_obs_space: bool = False
    order_id_counter_start_when_resetting: int = -200
    shuffle_action_messages: bool = True
    use_pickles_for_init: bool = True
    save_raw_observations: bool = False


@dataclass(frozen=True)
class MultiAgentConfig:
    world_config: World_EnvironmentConfig = World_EnvironmentConfig()
    dict_of_agents_configs: dict = field(default_factory=lambda: dict([
        ("MarketMaking", MarketMaking_EnvironmentConfig()),
        ("Execution", Execution_EnvironmentConfig()),
    ]))
    number_of_agents_per_type: list = field(default_factory=lambda: [1, 1])

    def __post_init__(self):
        for agent_type, config in self.dict_of_agents_configs.items():
            if "message" in config.observation_space:
                object.__setattr__(self.world_config, 'any_message_obs_space', True)


# Agent config type lookup (used by config_io autodetection)
CONFIG_OBJECT_DICT = {
    "MarketMaking": MarketMaking_EnvironmentConfig,
    "Execution": Execution_EnvironmentConfig,
}
