"""
GRPO Phase Training — 分层训练版本.

支持三个训练阶段:
  - "mm_only":  只训练做市商 (MM), 执行端(EXE)冻结
  - "exe_only": 只训练执行端 (EXE), 做市商(MM)冻结
  - "joint":    同时训练两者 (等同于 GRPO 原版)

在 make_train() 中读取 config["TRAIN_PHASE"] 决定阶段。
冻结的 agent 使用 identity optimizer, 参数不更新但环境交互正常。

Based on PureJaxRL Implementation of PPO — GRPO variant.
"""

import os

import pandas as pd
import csv

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import time
import jax # type: ignorepip 
jax.config.update('jax_disable_jit', False)
import flax
import jax.numpy as jnp # type: ignore
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal # type: ignore
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

#from jaxmarl.wrappers.baselines import SMAXLogWrapper
#from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.config.env_configs import MultiAgentConfig, Execution_EnvironmentConfig, World_EnvironmentConfig, MarketMaking_EnvironmentConfig

import wandb
import functools
import matplotlib.pyplot as plt

import sys

class ScannedLSTM(nn.Module):
    """LSTM cell scanned over time with episode-level reset.

    Wraps ``nn.LSTMCell`` inside a ``nn.scan`` so the same parameters
    are reused across all steps.  On episode boundaries (``resets==True``)
    the hidden/cell states are re-initialised to zeros.
    """

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module. carry = (hidden, cell) tuple for LSTM."""
        ins, resets = x
        # LSTM carry is (hidden, cell); reset both on episode done
        reset_state = self.initialize_carry(carry[0].shape[0], carry[0].shape[1])
        hidden = jnp.where(resets[:, jnp.newaxis], reset_state[0], carry[0])
        cell   = jnp.where(resets[:, jnp.newaxis], reset_state[1], carry[1])
        new_rnn_state, y = nn.LSTMCell(features=ins.shape[1])((hidden, cell), ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.LSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorOnlyRNN(nn.Module):
    """Actor-only recurrent policy network (no critic).

    Architecture:
        obs -> Dense(FC_DIM, relu) -> ScannedLSTM ->
            Dense(GRU_DIM, relu) -> Dense(action_dim) -> Categorical

    Attributes:
        action_dim: Number of discrete actions.
        config: Training config (must contain FC_DIM_SIZE, GRU_HIDDEN_DIM).
    """

    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        # obs, dones, avail_actions = x
        obs, dones = x

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)

        hidden, embedding = ScannedLSTM()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )

        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0) # type: ignore
        )(actor_mean)
        # Avail actions are not used in the current implementation, but can be added if needed.
        # unavail_actions = 1 - avail_actions
        action_logits = actor_mean # - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class GRPOTransition(NamedTuple):
    """Single transition step stored for GRPO training.

    Fields are flat (per-agent-type) to simplify minibatching.

    Attributes:
        global_done: ``done["__all__"]`` tiled to match agent count.
        done: Per-agent done flag (pre-step, for RNN reset).
        action: Action taken (shape: ``(num_actors,)``).
        reward: Reward received (shape: ``(num_actors,)``).
        log_prob: Log-prob under current policy.
        obs: Observation at start of step.
        info: Auxiliary info dict (world + agent keys).
    """
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # avail_actions: jnp.ndarray


def batchify(x: jnp.ndarray, num_actors: int) -> jnp.ndarray:
    """Reshape ``(num_envs, num_agents, ...)`` -> ``(num_actors, ...)``.

    Merges env and agent dims into a flat actor dimension.

    Args:
        x: Array with leading dim ``num_envs * num_agents``.
        num_actors: Target size (must equal product of first two dims).

    Returns:
        Array of shape ``(num_actors, -1)``.
    """
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, num_envs: int, num_agents: int) -> jnp.ndarray:
    """Reverse of :func:`batchify` — split flat actor dim back.

    Args:
        x: Array with leading dim ``num_envs * num_agents``.
        num_envs: Number of parallel environments.
        num_agents: Number of agents of this type per env.

    Returns:
        Array of shape ``(num_envs, num_agents, -1)``.
    """
    return x.reshape((num_envs, num_agents, -1))


def make_train(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
    init_key = jax.random.PRNGKey(config["SEED"])
    config_dict={"MarketMaking": MarketMaking_EnvironmentConfig,"Execution": Execution_EnvironmentConfig}
    print("init_key: ", init_key)
    ###############CLAUDE##############
    # Create a MultiAgentConfig object with parameters from the config
    # Build a case-insensitive field mapping for dataclass constructors.
    # YAML keys may differ in case from dataclass field names (e.g. "inventoryPnL_eta").
    def _build_agent_cfg(agent_type, agent_cfg):
        cls = config_dict[agent_type]
        valid_fields = {f.name.lower(): f.name for f in cls.__dataclass_fields__.values()}
        mapped = {}
        for k, v in agent_cfg.items():
            key_lower = k.lower()
            if key_lower in valid_fields:
                mapped[valid_fields[key_lower]] = v
            else:
                # pass through as-is for forward compatibility
                mapped[k] = v
        return cls(**mapped)

    agent_configs = {}
    if "AGENT_CONFIGS" in config:
        agent_configs = {
            agent_type: _build_agent_cfg(agent_type, agent_cfg)
            for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
        }
    else:
        agent_configs = {
            agent_type: config_dict[agent_type]()
            for agent_type, agent_cfg in config_dict.items()
        }
    print("agent_configs:", agent_configs)
    


    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            timePeriod=config["TimePeriod"],
            # Only override parameters that exist in both config and World_EnvironmentConfig
            **{k: v for k, v in config["world_config"].items() 
            if hasattr(World_EnvironmentConfig(), k) and k not in ["seed",
                                                                    "timePeriod",
                                                                    ]}
        ))
    print(ma_config)

    print("MultiAgentInventoryPenalty",ma_config.dict_of_agents_configs["MarketMaking"].inv_penalty)

    # For evaluation, create a separate config with evaluation-specific parameters
    eval_agent_configs = {}
    if "AGENT_CONFIGS" in config:
        eval_agent_configs = {
            agent_type: _build_agent_cfg(agent_type, agent_cfg)
            for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
        }
    else:
        eval_agent_configs = {
            agent_type: config_dict[agent_type]()
            for agent_type, agent_cfg in config_dict.items()
        }
        
    eval_ma_config = None
    if config["CALC_EVAL"]:
        eval_ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=eval_agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            timePeriod=config["EvalTimePeriod"],
            # Only override parameters that exist in both config and World_EnvironmentConfig
            **{k: v for k, v in config["world_config"].items() 
            if hasattr(World_EnvironmentConfig(), k) and k not in ["seed",
                                                                    "timePeriod",
                                                                    ]}
        ))
   



    env : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config)
    if config["CALC_EVAL"]:
        eval_env: MARLEnv = MARLEnv(key=init_key,multi_agent_config=eval_ma_config)
    else:
        eval_env = None
    
    agent_type_names = list(env.type_names)

    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZES"] = [
        nact * config["NUM_STEPS"] // config["NUM_MINIBATCHES"] for i,nact in enumerate(config["NUM_ACTORS_PERTYPE"])
    ]
    # config["CLIP_EPS"] = (
    #     config["CLIP_EPS"] / env.num_agents
    #     if config["SCALE_CLIP_EPS"]
    #     else config["CLIP_EPS"]
    # )

    def linear_schedule(lr,count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    def train(rng):
        # INIT NETWORK


        # For a given agent type (instance) we need the following inputs:
        # Action space, obs space, 

        # The outputs that depends on these and are kept seperate are;
        # - network, init_x, init_hstate, network_params, train_state
        hstates = []
        network_params_list = []
        train_states = []
        num_agents_of_instance_list = []
        init_dones_agents = []
        for i, instance in enumerate(env.instance_list):
            # print("Action space dimension for network i ",env.action_spaces[i].n)
            network = ActorOnlyRNN(env.action_spaces[i].n, config=config)
            rng, _rng = jax.random.split(rng)
            init_x = (
                jnp.zeros(
                    (1, config["NUM_ENVS"], env.observation_spaces[i].shape[0])
                ), # obs
                jnp.zeros((1, config["NUM_ENVS"])), # dones
                # jnp.zeros((1, config["NUM_ENVS"], env.action_spaces[i].n)), #     avail_actions
            )

            # FIXME: very unsure about this, why is it NUM_ENVS and not NUM_ACTORS?
            init_hstate = ScannedLSTM.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            network_params = network.init(_rng, init_hstate, init_x)
            if config["ANNEAL_LR"][i]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(learning_rate=functools.partial(linear_schedule,config["LR"][i]), eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(config["LR"][i], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            init_hstate = ScannedLSTM.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"])

            # Instead of appending dicts, maintain separate lists for each attribute
            hstates.append(init_hstate)
            network_params_list.append(network_params)
            train_states.append(train_state)
            num_agents_of_instance_list.append(env.multi_agent_config.number_of_agents_per_type[i])
            init_dones_agents.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))

        # ========== Phase Training Logic ==========
        phase = config.get("TRAIN_PHASE", "joint")
        agent_type_names = list(env.type_names)
        # type_names 使用 short_name，如 ["MM", "EXE"]

        # Phase 2: 先加载 checkpoint，再冻结（防止 .replace() 覆盖 frozen_tx）
        if phase == "exe_only" and "CHECKPOINT_PATH" in config:
            ckpt_path = config["CHECKPOINT_PATH"]
            print(f"  Loading checkpoint for MM from: {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                restored = flax.serialization.from_bytes(
                    train_states[0].params, f.read()
                )
            train_states[0] = train_states[0].replace(params=restored)
            print("  Checkpoint loaded.")

        # Freeze agents based on phase
        for i in range(len(train_states)):
            should_update = True
            if phase == "mm_only" and agent_type_names[i] != "MM":
                should_update = False
            elif phase == "exe_only" and agent_type_names[i] != "EXE":
                should_update = False

            if should_update:
                print(f"  Training agent: {agent_type_names[i]}")
            else:
                print(f"  Freezing agent: {agent_type_names[i]}")
                frozen_tx = optax.chain(
                    optax.set_to_zero(),
                    optax.sgd(learning_rate=0.0),
                )
                train_states[i] = train_states[i].replace(tx=frozen_tx)
        # ==========================================

        train_states=flax.jax_utils.replicate(train_states)
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_params=env.default_params
        if config["CALC_EVAL"]:
            eval_env_params=eval_env.default_params # type: ignore
        else:
            eval_env_params = None

        # env_params=jax.device_put(env_params)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng,env_params)
        # TRAIN LOOP

        def reshape_pytree_leading_dim(pytree, devices):
            """
            Reshape each leaf of a pytree by splitting the leading dimension M into (N, M/N).
            
            Args:
                pytree: A PyTree with arrays that have a leading dimension to reshape.
                num_envs: The number of environments (N) for the first dimension.
                num_agents: Optional, the number of agents per environment. If None, 
                           calculated as M / num_envs.
            
            Returns:
                A PyTree with the same structure but with leaves reshaped from (M, ...) 
                to (N, M/N, ...).
            """
            def _reshape_leaf(leaf):
                if not isinstance(leaf, jnp.ndarray):
                    return leaf
                
                if leaf.ndim == 0:
                    return leaf
                
                leading_dim = leaf.shape[0]
                # If num_agents is provided, use it; otherwise calculate it
                env_per_device = leading_dim // devices
                
                # Check that the reshape is valid
                if leading_dim != devices * env_per_device:
                    raise ValueError(
                        f"Leading dimension {leading_dim} cannot be reshaped to "
                        f"({devices}, {env_per_device})"
                    )
                    
                new_shape = (devices, env_per_device) + leaf.shape[1:]
                return leaf.reshape(new_shape)
            
            return jax.tree_util.tree_map(_reshape_leaf, pytree)

        env_state=reshape_pytree_leading_dim(env_state, config["N_DEVICES"])
        obsv=reshape_pytree_leading_dim(obsv, config["N_DEVICES"])
        init_dones_agents=reshape_pytree_leading_dim(init_dones_agents, config["N_DEVICES"]) # last_done
        hstates=reshape_pytree_leading_dim(hstates, config["N_DEVICES"])

        print(jax.tree_util.tree_map(lambda x: x.shape, (env_state,obsv,init_dones_agents,hstates,_rng)))


        def callback(metric):
            print("Update step:", metric["update_steps"])
            action_distribution = {}
            for i, tr in enumerate(metric["traj_batch"]):
                actions = np.array(tr.action).flatten()
                unique_actions, counts = np.unique(actions, return_counts=True)
                tot_counts=sum(counts)
                # Add each action count to the dictionary with a unique key
                for a, c in zip(unique_actions, counts):
                    action_distribution[f"action_{i}_{int(a)}"] = c/tot_counts*100
            logging_dict = {
                    # TODO: Log the quantities of interest. Keep it trivial for now.
                    "env_step": (metric["update_steps"].sum()+1)
                    * config["NUM_ENVS"]// config["N_DEVICES"]
                    * config["NUM_STEPS"],
                    **{f"network_{i}": m for i,m in enumerate(metric["loss"])},
                    **{f"avg_reward_{i}": metric["avg_reward"][i].mean() for i in range(len(metric["avg_reward"]))},
                    **action_distribution
                }
            if config["CALC_EVAL"]:
                logging_dict.update({
                    **{f"avg_eval_reward_{i}": metric["avg_reward_eval"][i].mean() for i in range(len(metric["avg_reward_eval"]))},
                })
            if config["WANDB_MODE"]!= "disabled":
                wandb.log(logging_dict)

            for i in range(len(metric["avg_reward"])):
                print(f"avg_reward_{i} {metric['avg_reward'][i]}")
                if config["CALC_EVAL"]:
                    print(f"avg_eval_reward_{i} {metric['avg_reward_eval'][i]}")
        def speed_only_callback(metric):
            logging_dict = {
                    "env_step": (metric["update_steps"][0]+1)
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]}
            print(metric["update_steps"],config["NUM_ENVS"],config["NUM_STEPS"])
            print(logging_dict["env_step"])
            if config["WANDB_MODE"]!= "disabled":
                wandb.log(logging_dict) 


        def _update_step(update_runner_state,env_params,eval_env_params):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done,h_states, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Ignore getting the available actions for now, assume all actions are available.
                # avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                # avail_actions = jax.lax.stop_gradient(
                #     batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                # )
                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actions=[]
                log_probs=[]

                for i, train_state in enumerate(train_states):
                    rng_, _rng = jax.random.split(_rng)
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])  # Reshape to match the input shape of the network
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    h_states[i], pi = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                    action = pi.sample(seed=_rng)
                    log_probs.append(pi.log_prob(action))
                    action=unbatchify(action, config["NUM_ENVS"]//config["N_DEVICES"], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                    actions.append(action.squeeze())
                    # env_act = unbatchify(
                    #     action, env.agents, config["NUM_ENVS"], env.num_agents
                    # )
                    # env_act = {k: v.squeeze() for k, v in env_act.items()}
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"]//config["N_DEVICES"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0,None)
                )(rng_step, env_state, actions,env_params)

                # info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                
                done_batch=done
                transitions=[]
                for i,train_state in enumerate(train_states):
                    done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze()
                    obs_batch = batchify(last_obs[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                    action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                    log_prob = log_probs[i]

                    info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]),info["agents"][i])}
                    # print(f"info for agenttype {i}:", info_i)


                    transitions.append(GRPOTransition(
                        jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                        last_done[i],
                        action_batch.squeeze(),
                        batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze(),
                        log_prob.squeeze(),
                        obs_batch,
                        info_i,
                        # avail_actions,
                    ))
                runner_state = (train_states, env_state, obsv, done_batch['agents'], h_states, rng)
                return runner_state, transitions

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )


            

            # CALCULATE GROUP ADVANTAGE (GRPO) — episode-level
            # Normalize total episode returns across agents, NOT per-step rewards.
            # This preserves GRPO's core idea: compare full trajectories, not instant rewards.
            train_states, env_state, last_obs, last_dones, hstates_new, rng = runner_state

            advantages = []
            ref_params_list = []
            for i in range(len(train_states)):
                # Episode return per agent: sum over all NUM_STEPS
                # traj_batch[i].reward: (NUM_STEPS, NUM_AGENTS)
                episode_returns = jnp.sum(traj_batch[i].reward, axis=0)  # (NUM_AGENTS,)
                mean_r = jnp.mean(episode_returns)
                std_r = jnp.std(episode_returns) + 1e-8
                # Scalar advantage per agent — same for every step in that agent's trajectory
                adv_per_agent = (episode_returns - mean_r) / std_r  # (NUM_AGENTS,)
                advantages_i = adv_per_agent[jnp.newaxis, :]  # (1, NUM_AGENTS) — broadcasts over steps
                advantages.append(advantages_i)
                # Snapshot reference policy BEFORE any minibatch/epoch updates
                ref_params_list.append(train_states[i].params)
            # UPDATE NETWORKS
            loss_infos = []
            for i, train_state in enumerate(train_states):
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages = batch_info
                        # ref_params captured from _update_epoch scope (fixed for all epochs)

                        def _loss_fn(params, init_hstate, traj_batch, advantages, ref_params):
                            # RERUN NETWORK (actor only, no critic)
                            _, pi = train_state.apply_fn(
                                params,
                                (init_hstate[0].squeeze(), init_hstate[1].squeeze()),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob = pi.log_prob(traj_batch.action)

                            # Reference policy for KL penalty
                            _, pi_ref = train_state.apply_fn(
                                ref_params,
                                (init_hstate[0].squeeze(), init_hstate[1].squeeze()),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob_ref = pi_ref.log_prob(traj_batch.action)

                            # CALCULATE ACTOR LOSS (PPO clip)
                            logratio = log_prob - traj_batch.log_prob
                            ratio = jnp.exp(logratio)
                            loss_actor1 = ratio * advantages
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * advantages
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().mean()

                            # KL penalty (GRPO): (pi_ref/pi) - log(pi_ref/pi) - 1
                            lograt = log_prob_ref - log_prob
                            kl_penalty = jnp.mean(jnp.exp(lograt) - lograt - 1.0)

                            # debug
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                            total_loss = (
                                loss_actor
                                + config["KL_COEF"][i] * kl_penalty
                                - config["ENT_COEF"][i] * entropy
                            )
                            return total_loss, (kl_penalty, loss_actor, entropy, ratio, approx_kl, clip_frac)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, init_hstate, traj_batch, advantages, ref_params
                        )
                        total_loss=jax.lax.pmean(total_loss, axis_name="device_batch")
                        grads = jax.lax.pmean(grads, axis_name="device_batch")
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss
                    (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        rng,
                        ref_params,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    # adding an additional "fake" dimensionality to perform minibatching correctly
                    init_hstate = (
                        jnp.reshape(init_hstate[0], (1, config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"], -1)),
                        jnp.reshape(init_hstate[1], (1, config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"], -1)),
                    )
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages,
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])

                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        (init_hstate[0].squeeze(), init_hstate[1].squeeze()),
                        traj_batch,
                        advantages,
                        rng,
                        ref_params,
                    )
                    return update_state, total_loss

                update_state = (
                    train_state,
                    initial_hstates[i],
                    traj_batch[i],
                    advantages[i],
                    rng,
                    ref_params_list[i],  # fixed ref policy for all epochs in this update step
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_states[i] = update_state[0]
                loss_infos.append(loss_info)



            metrics= {}
            metrics['agents'] = [jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"]//config["N_DEVICES"], config["NUM_AGENTS_PER_TYPE"][i])
                ),
                trjbtch.info['agent']) for i, trjbtch in enumerate(traj_batch)]
            metrics['world'] = [traj_batch.info['world'] for i, traj_batch in enumerate(traj_batch)]
            metrics["loss"]=[]
            for i,loss_info in enumerate(loss_infos):
                ratio_0 = loss_info[1][3].at[0,0].get().mean()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
                metrics["loss"].append({
                    "total_loss": loss_info[0],
                    "kl_penalty": loss_info[1][0],
                    "actor_loss": loss_info[1][1],
                    "entropy": loss_info[1][2],
                    "ratio": loss_info[1][3],
                    "ratio_0": ratio_0,
                    "approx_kl": loss_info[1][4],
                    "clip_frac": loss_info[1][5],
                    "weighted_entropy_loss": loss_info[1][2] * config["ENT_COEF"][i],
                })
            metrics['avg_reward'] = [jnp.mean(tr.reward) for tr in traj_batch]
            metrics["traj_batch"] = traj_batch



            if config["CALC_EVAL"]:
                def _eval_step(eval_runner_state, unused):
                    train_states, eval_env_state, last_obs, last_done,hstates, rng = eval_runner_state
                    rng, _rng = jax.random.split(rng)
                
                    actions=[]
                    log_probs=[]

                    for i, train_state in enumerate(train_states):
                        obs_i= last_obs[i]
                        obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])  # Reshape to match the input shape of the network
                        ac_in = (
                            obs_i[jnp.newaxis, :],
                            last_done[i][jnp.newaxis, :],
                            # avail_actions,
                        )
                        hstates[i], pi, value = train_state.apply_fn(train_state.params, hstates[i], ac_in)
                        action = pi.sample(seed=_rng)
                        log_probs.append(pi.log_prob(action))
                        action=unbatchify(action, config["NUM_ENVS"]//config["N_DEVICES"], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                        actions.append(action.squeeze())

                        rng, _rng = jax.random.split(rng)

                





                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"]//config["N_DEVICES"])
                    obsv, eval_env_state, reward, done, info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0, None) # type: ignore
                    )(rng_step, eval_env_state, actions, eval_env_params)
                    done_batch=done
                    transitions=[]    

                    for i,train_state in enumerate(train_states):
                        done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze()
                        obs_batch = batchify(last_obs[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                        action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                        log_prob = log_probs[i]

                        info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]),info["agents"][i])}
                        # print(f"info for agenttype {i}:", info_i)


                        transitions.append(GRPOTransition(
                            jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                            last_done[i],
                            action_batch.squeeze(),
                            batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze(),
                            log_prob.squeeze(),
                            obs_batch,
                            info_i,
                            # avail_actions,
                        ))
                    eval_runner_state = (train_states, eval_env_state, obsv, done_batch['agents'], hstates, rng)
                    return eval_runner_state, transitions

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(rng, config["NUM_ENVS"]//config["N_DEVICES"])
                eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params) # type: ignore
                jax.debug.print("WINDOW INDECES: {}",eval_env_state.world_state.window_index)

                eval_hstates=[]
                init_dones_agents_eval=[]
                for i,train_state in enumerate(train_states):
                    eval_hstates.append(ScannedLSTM.initialize_carry(config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]), dtype=bool))


                
                eval_runner_state = (
                train_states,
                eval_env_state,
                eval_obsv,
                init_dones_agents_eval,
                eval_hstates,
                _rng,
                )
                eval_runner_state, eval_traj_batch = jax.lax.scan(
                    _eval_step, eval_runner_state, None,  config["NUM_STEPS_EVAL"]
                )
                metrics['agents_eval'] = [jax.tree.map(
                    lambda x: x.reshape(
                        (config["NUM_STEPS_EVAL"], config["NUM_ENVS"]//config["N_DEVICES"], config["NUM_AGENTS_PER_TYPE"][i])
                    ),
                    trjbtch.info['agent']) for i, trjbtch in enumerate(eval_traj_batch)]
                metrics['world_eval'] = [trjbtch.info['world'] for i, trjbtch in enumerate(eval_traj_batch)]
                if config["CALC_EVAL"]:
                    metrics['avg_reward_eval'] = [jnp.mean(tr.reward) for tr in eval_traj_batch]
                    metrics["traj_batch_eval"] = eval_traj_batch

            
            metrics["update_steps"] = update_steps
            # jax.experimental.io_callback(callback, None, metrics)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_dones, hstates_new, rng)

            # jax.profiler.save_device_memory_profile(f"memory_{update_steps}.prof")
            return (runner_state, update_steps), metrics

        rng, _rng = jax.random.split(rng)
        device_rng=jax.random.split(_rng, config["N_DEVICES"])  # Split the RNG for each device
        runner_state = (
            train_states,
            env_state,
            obsv,
            init_dones_agents, # last_done
            hstates,  # initial hidden states for RNN
            device_rng,
        )


        jitted_update_step = jax.jit(_update_step,)
        pmapped_update_step = jax.pmap(
            jitted_update_step,
            axis_name="device_batch",
            in_axes=(((0, 0, 0, 0, 0, 0),None),None,None),
            out_axes=(((0, 0, 0, 0, 0, 0),None), 0),
        )
        updates=0
        # compiled_update_step = jax.jit(pmapped_update_step).trace((runner_state,updates),env_params,eval_env_params).lower().compile()  # type: ignore



        # Print details about the runner state components before training starts
        print("========== Runner State Components ==========")
        print(f"Number of train states: {len(train_states)}")
        for i, ts in enumerate(train_states):
            print(f"Train state {i} structure: {jax.tree_util.tree_structure(ts)}")

        print(f"\nEnvironment state structure: {jax.tree_util.tree_structure(env_state)}")

        print("\nObservation shapes:")
        for i, obs in enumerate(obsv):
            print(f"Agent type {i} observation shape: {obs.shape}")

        print("\nInitial done flags:")
        for i, done in enumerate(init_dones_agents):
            print(f"Agent type {i} done flags shape: {done.shape}, dtype: {done.dtype}")

        print("\nHidden state details:")
        for i, h in enumerate(hstates):
            if isinstance(h, tuple):
                print(f"Agent type {i} hidden (h) shape: {h[0].shape}, cell (c) shape: {h[1].shape}")
            else:
                print(f"Agent type {i} hidden state shape: {h.shape}")

        
        for i in range(config["NUM_UPDATES"]):
            print(f"Update step {i+1}/{config['NUM_UPDATES']}")
            # Run the update step:
            # if i>2 and i<4:
            #     jax.profiler.start_trace("/tmp/profile-data")
            (runner_state,updates),metrics=pmapped_update_step((runner_state,updates),env_params,eval_env_params)
            callback(metrics)

            # if i>2 and i<4:
            #     jax.block_until_ready((runner_state,updates,metrics))
            #     jax.profiler.stop_trace()


            # callback(metrics)
            

            
            del metrics
            gc.collect()
        


        # runner_state, metrics = jax.lax.scan(
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )
        
        
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base="1.3", config_path="../../../config/rl_configs", config_name="PMAP_ippo_rnn_JAXMARL_2player")
def main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    # Save YAML world_config before merge (env_config defaults override it)
    yaml_world_config = config.get("world_config", {})
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)
    # Restore YAML world_config
    config["world_config"] = yaml_world_config

    print(config)

    def sweep_fun():
        print(f"WANDB CONFIG PRIOR {wandb.config}")


        run=wandb.init(
            entity=config["ENTITY"], # type: ignore
            project=config["PROJECT"], # type: ignore
            tags=["IPPO", "RNN"], # type: ignore
            config=config, # type: ignore
            mode=config["WANDB_MODE"], # type: ignore
            allow_val_change=True,
        )
        # params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        
        
        # print(f"WANDB CONFIG {wandb.config}")
        # +++++ Single GPU +++++
        

        rng = jax.random.PRNGKey(config["SEED"])

        print("wandb.config", wandb.config)

        if config["Timing"]:
            start_time = time.time()


        train_fun = make_train(config)
        out = train_fun(rng)
        # train_state = out['runner_state'][0] # runner_state.train_state
        # params = train_state.params

        if config["Timing"]:
            end_time = time.time()
            elapsed = end_time - start_time
            total_steps = config["TOTAL_TIMESTEPS"]
            agents_per_type = config["NUM_AGENTS_PER_TYPE"]
            num_data_msgs = config.get("n_data_msg_per_step", None)
            num_envs = config["NUM_ENVS"]

            # Print results
            print(f"Total steps: {total_steps}")
            print(f"Elapsed time: {elapsed} seconds")
            print(f"Steps per second: {total_steps / elapsed}")
            print(f"Agents per type: {agents_per_type}")
            print(f"Num data messages: {num_data_msgs}")
            print(f"Num envs: {num_envs}")

            # Save to CSV
            results = {
                "total_steps": [total_steps],
                "elapsed_seconds": [elapsed],
                "steps_per_second": [total_steps / elapsed],
                "agents_per_type": [str(agents_per_type)],
                "num_data_msgs": [num_data_msgs],
                "num_envs": [num_envs],
            }
            # df = pd.DataFrame(results)
            # csv_path = "timing_results.csv"
            # # Append if file exists, else write header
            # try:
            #     with open(csv_path, "x", newline="") as f:
            #         df.to_csv(f, index=False)
            # except FileExistsError:
            #     with open(csv_path, "a", newline="") as f:
            #         df.to_csv(f, index=False, header=False)

        
        # # Save the params to a file using flax.serialization.to_bytes
        # with open(params_file_name, 'wb') as f:
        #     f.write(flax.serialization.to_bytes(params))
        #     print(f"params saved")

        # Load the params from the file using flax.serialization.from_bytes
        # with open(params_file_name, 'rb') as f:
        #     restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        #     print(f"params restored")

        run.finish()

    # NOTE: Sweep Parameters will override the config file, but cannot be used to override any environment params currently. 
    # This latter option will require some careful thought on how best to implement - due to to variable number of agent types.
    sweep_parameters = {
        "LR": {"values": [config["LR"]]},
        #"GAMMA": {"values": [config["GAMMA"], [0.99,0.99]]},
        #"LR": {"values": [config["LR"], [0.004,0.004], [0.00004,0.00004]]},
        #"ENT_COEF": {"values": [config["ENT_COEF"], [0.1,0.1], [0.05,0.05]]},
        #"NUM_STEPS": {"values": [config["NUM_STEPS"], 2048 ,512]},
        #"CLIP_EPS": {"values": [config["CLIP_EPS"], 0.3, 0.1]},
        #"VF_COEF": {"values": [config["VF_COEF"], [1e-6,1e-7], [1e-9,1e-8]]},
        #"FC_DIM_SIZE": {"values": [config["FC_DIM_SIZE"], 256]},
       # "NUM_AGENTS_PER_TYPE": {"values": [config["NUM_AGENTS_PER_TYPE"], [2,2], [10,10]]},
       #"SEED": {"values": [2,3,4,5,6,7,8,9,10]},
       #"NUM_ENVS": {"values": [config["NUM_ENVS"]]},
       #"NUM_STEPS": {"values": [config["NUM_STEPS"], 128, 32, 8]},
       
        
        # "env_params" : {"parameters": {
        #                 "world_params" : {"parameters":
        #                                 {"n_data_msg_per_step": {"values":[50,150]},
        #                                 }
        #                                 },
        #                 }},
    }

    sweep_config={
        "method": "grid",
        "parameters": sweep_parameters,
    }
    print(sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project=config["PROJECT"],entity=config["ENTITY"])
    print(sweep_id)
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)

@hydra.main(version_base="1.3", config_path="../../../config/rl_configs", config_name="PMAP_ippo_rnn_JAXMARL_2player")
def seperate_main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    # Save YAML world_config before merge (env_config defaults override it)
    yaml_world_config = config.get("world_config", {})
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)
    # Restore YAML world_config
    config["world_config"] = yaml_world_config

    # Init wandb before running training
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["GRPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"],
        allow_val_change=True,
    )

    rng = jax.random.PRNGKey(0)

    train_fun = make_train(config)
    out = train_fun(rng)

    # Save checkpoint (for Phase 1 -> Phase 2 handoff)
    if config.get("SAVE_CHECKPOINT", False):
        import datetime
        runner_state = out.get("runner_state")
        if runner_state is not None:
            train_states = runner_state[0]
            # Un-replicate: take first device's copy
            train_state_0 = flax.jax_utils.unreplicate(train_states[0])
            ts = datetime.datetime.now().strftime("%m-%d_%H-%M")
            ckpt_name = f'params_phase1_{config["PROJECT"]}_{ts}.pkl'
            with open(ckpt_name, 'wb') as f:
                f.write(flax.serialization.to_bytes(train_state_0.params))
            print(f"Checkpoint saved: {ckpt_name}")

    # out=jax.block_until_ready(out)
    # (dummy * dummy).block_until_ready()
    # jax.profiler.stop_trace()


if __name__ == "__main__":
    seperate_main()