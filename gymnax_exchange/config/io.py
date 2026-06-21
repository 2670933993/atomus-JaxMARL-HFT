"""
Helper functions for saving and loading configuration objects to/from files.

Migrated from gymnax_exchange/jaxob/config_io.py.
"""

import json
import os
from typing import Dict, Any, Union
from dataclasses import asdict, fields

from gymnax_exchange.config.env_configs import (
    MultiAgentConfig,
    World_EnvironmentConfig,
    MarketMaking_EnvironmentConfig,
    Execution_EnvironmentConfig,
    JAXLOB_Configuration,
)


def save_config_to_file(config: MultiAgentConfig, filepath: str) -> None:
    """Save a MultiAgentConfig instance to a JSON file."""
    config_dict = asdict(config)

    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_file(filepath: str) -> MultiAgentConfig:
    """Load a MultiAgentConfig instance from a JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return _dict_to_multiagent_config(config_dict)


def _dict_to_multiagent_config(config_dict: Dict[str, Any]) -> MultiAgentConfig:
    """Convert a dictionary to a MultiAgentConfig instance."""
    world_config_dict = config_dict.get('world_config', {})
    world_config = _dict_to_world_config(world_config_dict)

    agents_configs_dict = config_dict.get('dict_of_agents_configs', {})
    dict_of_agents_configs = {}

    for agent_type, agent_config_dict in agents_configs_dict.items():
        if agent_type == "MarketMaking":
            dict_of_agents_configs[agent_type] = _dict_to_marketmaking_config(agent_config_dict)
        elif agent_type == "Execution":
            dict_of_agents_configs[agent_type] = _dict_to_execution_config(agent_config_dict)
        else:
            dict_of_agents_configs[agent_type] = _auto_detect_agent_config(agent_config_dict)

    number_of_agents_per_type = config_dict.get('number_of_agents_per_type', [1])

    return MultiAgentConfig(
        world_config=world_config,
        dict_of_agents_configs=dict_of_agents_configs,
        number_of_agents_per_type=number_of_agents_per_type,
    )


def _dict_to_world_config(config_dict: Dict[str, Any]) -> World_EnvironmentConfig:
    """Convert a dictionary to a World_EnvironmentConfig, filling missing values from defaults."""
    default_config = World_EnvironmentConfig()
    kwargs = {}
    for field in fields(World_EnvironmentConfig):
        val = config_dict.get(field.name, getattr(default_config, field.name))
        # Expand ~ in path fields so JSON placeholders like "~" work portably
        if field.name in ("alphatradePath", "dataPath") and isinstance(val, str):
            val = os.path.expanduser(val)
        kwargs[field.name] = val
    return World_EnvironmentConfig(**kwargs)


def _dict_to_marketmaking_config(config_dict: Dict[str, Any]) -> MarketMaking_EnvironmentConfig:
    """Convert a dictionary to a MarketMaking_EnvironmentConfig, filling missing values from defaults."""
    default_config = MarketMaking_EnvironmentConfig()
    kwargs = {}
    for field in fields(MarketMaking_EnvironmentConfig):
        kwargs[field.name] = config_dict.get(field.name, getattr(default_config, field.name))
    return MarketMaking_EnvironmentConfig(**kwargs)


def _dict_to_execution_config(config_dict: Dict[str, Any]) -> Execution_EnvironmentConfig:
    """Convert a dictionary to an Execution_EnvironmentConfig, filling missing values from defaults."""
    default_config = Execution_EnvironmentConfig()
    kwargs = {}
    for field in fields(Execution_EnvironmentConfig):
        kwargs[field.name] = config_dict.get(field.name, getattr(default_config, field.name))
    return Execution_EnvironmentConfig(**kwargs)


def _auto_detect_agent_config(
    config_dict: Dict[str, Any],
) -> Union[MarketMaking_EnvironmentConfig, Execution_EnvironmentConfig]:
    """Auto-detect agent config type by field overlap."""
    mm_fields = set(f.name for f in fields(MarketMaking_EnvironmentConfig))
    exec_fields = set(f.name for f in fields(Execution_EnvironmentConfig))
    config_keys = set(config_dict.keys())
    mm_overlap = len(config_keys.intersection(mm_fields))
    exec_overlap = len(config_keys.intersection(exec_fields))
    if mm_overlap >= exec_overlap:
        return _dict_to_marketmaking_config(config_dict)
    return _dict_to_execution_config(config_dict)


def save_config_to_yaml(config: MultiAgentConfig, filepath: str) -> None:
    """Save a MultiAgentConfig instance to a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install PyYAML")

    config_dict = asdict(config)

    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config_from_yaml(filepath: str) -> MultiAgentConfig:
    """Load a MultiAgentConfig instance from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install PyYAML")

    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)

    return _dict_to_multiagent_config(config_dict)


def get_config_summary(config: MultiAgentConfig) -> str:
    """Get a human-readable summary of the configuration."""
    summary_lines = []
    summary_lines.append("=== MultiAgent Configuration Summary ===")
    summary_lines.append("")
    summary_lines.append("World Configuration:")
    summary_lines.append(f"  Episode type: {config.world_config.ep_type}")
    summary_lines.append(f"  Episode time: {config.world_config.episode_time}")
    summary_lines.append(f"  Stock: {config.world_config.stock}")
    summary_lines.append(f"  Data messages per step: {config.world_config.n_data_msg_per_step}")
    summary_lines.append("")
    summary_lines.append("Agent Configurations:")
    for agent_type, agent_config in config.dict_of_agents_configs.items():
        summary_lines.append(f"  {agent_type}:")
        if hasattr(agent_config, 'action_space'):
            summary_lines.append(f"    Action space: {agent_config.action_space}")
        if hasattr(agent_config, 'observation_space'):
            summary_lines.append(f"    Observation space: {agent_config.observation_space}")
        if hasattr(agent_config, 'reward_space'):
            summary_lines.append(f"    Reward space: {agent_config.reward_space}")
        if hasattr(agent_config, 'n_actions'):
            summary_lines.append(f"    Number of actions: {agent_config.n_actions}")
    summary_lines.append("")
    summary_lines.append(f"Number of agents per type: {config.number_of_agents_per_type}")
    return "\n".join(summary_lines)
