"""
[BACKWARD-COMPATIBLE SHIM] — imports from gymnax_exchange.config.env_configs.

All environment configuration dataclasses have moved to
gymnax_exchange/config/env_configs.py.

This file re-exports everything for existing code that imports from here.
"""

from gymnax_exchange.config.env_configs import (            # noqa: F401, F403
    JAXLOB_Configuration,
    MarketMaking_EnvironmentConfig,
    Execution_EnvironmentConfig,
    World_EnvironmentConfig,
    MultiAgentConfig,
    CONFIG_OBJECT_DICT,
)
