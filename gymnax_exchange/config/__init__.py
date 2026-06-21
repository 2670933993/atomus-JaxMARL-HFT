"""
JaxMARL-HFT Configuration Package.

Centralized location for constants, environment configurations, and I/O utilities.

## Quick Reference

| Module | Contents |
|--------|----------|
| `constants` | Enum constants (MaxInt, MessageType, CancelMode, ...) and magic values |
| `env_configs` | Dataclass configs (JAXLOB_Configuration, MarketMaking/Execution/World/MultiAgent configs) |
| `io` | JSON/YAML serialization helpers |

See individual modules for details.
"""

from gymnax_exchange.config.constants import *

# env_configs and io are loaded lazily / explicitly when needed.
# Import them directly:
#   from gymnax_exchange.config.env_configs import MultiAgentConfig
#   from gymnax_exchange.config.io import load_config_from_file
