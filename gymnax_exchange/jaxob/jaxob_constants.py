"""
[BACKWARD-COMPATIBLE SHIM] — imports from gymnax_exchange.config.constants.

All constants have been consolidated into gymnax_exchange/config/constants.py.
This file re-exports everything for existing code that imports from here.
"""

from gymnax_exchange.config.constants import *  # noqa: F401, F403
