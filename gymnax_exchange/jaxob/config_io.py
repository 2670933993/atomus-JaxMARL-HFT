"""
[BACKWARD-COMPATIBLE SHIM] — imports from gymnax_exchange.config.io.

All config I/O helpers have moved to gymnax_exchange/config/io.py.
This file re-exports them for existing code that imports from here.
"""

from gymnax_exchange.config.io import (               # noqa: F401, F403
    save_config_to_file,
    load_config_from_file,
    save_config_to_yaml,
    load_config_from_yaml,
    get_config_summary,
)
