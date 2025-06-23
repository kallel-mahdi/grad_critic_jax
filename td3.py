from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_map
import distrax # Though not strictly needed for TD3 policy, keep for potential future distribution use

# Import common components from utils and base_agent
from utils import Batch, MLP, default_init, PRNGKey, Params, InfoDict
from base_agent import RLAgent, RLAgentState, RLAgentConfig
# Import networks
from networks import  DoubleCritic, DeterministicActor

@struct.dataclass
class TD3Config(RLAgentConfig):
    actor_lr: float
    critic_lr: float
    hidden_dims: Sequence[int]
    discount: float
    tau: float
    policy_noise: float
    noise_clip: float
    policy_delay: int
    exploration_noise: float
    max_action: float
    final_fc_init_scale: float

@struct.dataclass
class TD3State(RLAgentState): # Inherit from RLAgentState
    rng: PRNGKey
    step: int
    actor: train_state.TrainState
    critic: train_state.TrainState # Holds parameters for both Q networks
    target_actor_params: Params
    target_critic_params: Params
    config: TD3Config



# --- TD3 Critic Update ---
def update_critic(key_critic: PRNGKey, state: TD3State, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:

    # Target Policy Smoothing: Add noise to target actions
    noise = (jax.random.normal(key_critic, batch.actions.shape) * state.config.policy_noise
            ).clip(-state.config.noise_clip, state.config.noise_clip)

    # Compute next actions using the target actor network and add clipped noise
    next_actions = (
        state.actor.apply_fn({'params': state.target_actor_params}, batch.next_observations) + noise
    ).clip(-state.config.max_action, state.config.max_action) # Clip actions to valid range

    # Compute target Q-values using the target critic parameters (DoubleCritic).
    next_q1, next_q2 = state.critic.apply_fn({'params': state.target_critic_params}, batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    # Bellman target: reward + discounted future Q-value.
    target_q = batch.rewards + state.config.discount * batch.masks * next_q
    target_q = jax.lax.stop_gradient(target_q) # Stop gradient flow through the target

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Compute current Q-values using the main critic network.
        q1, q2 = state.critic.apply_fn({'params': critic_params}, batch.observations, batch.actions)

        # Critic loss: Mean Squared Bellman Error (MSBE) for both critics.
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    # Compute gradients and update critic model using TrainState's apply_gradients
    grads, info = jax.grad(critic_loss_fn, has_aux=True)(state.critic.params)
    new_critic = state.critic.apply_gradients(grads=grads)

    return new_critic, info


# --- TD3 Actor Update ---
def update_actor(state: TD3State, batch: Batch) -> Tuple[train_state.TrainState, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Apply actor model to get actions
        actions = state.actor.apply_fn({'params': actor_params}, batch.observations)

        # Evaluate actions using *one* of the critic networks (typically the first one).
        # The critic parameters used here are the *updated* ones from the critic step.
        q1, _ = state.critic.apply_fn({'params': state.critic.params}, batch.observations, actions)

        # Actor loss: maximize Q-value of the generated actions.
        actor_loss = -q1.mean()

        return actor_loss, {'actor_loss': actor_loss}

    # Compute gradients and update actor model
    grads, info = jax.grad(actor_loss_fn, has_aux=True)(state.actor.params)
    new_actor = state.actor.apply_gradients(grads=grads)

    return new_actor, info


# --- Target Network Update ---
def target_update(params: Params, target_params: Params, tau: float) -> Params:
    """Soft update for target networks."""
    new_target_params = tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), params, target_params)
    return new_target_params


# --- TD3 Update Steps (External JIT functions) ---
@jax.jit
def _td3_update_step(state: TD3State, batch: Batch) -> Tuple[TD3State, InfoDict]:
    """Single TD3 update step (external JIT)."""
    rng, key_critic = jax.random.split(state.rng)

    # Update critic first
    new_critic, critic_info = update_critic(key_critic, state, batch)
    state = state.replace(critic=new_critic)

    # Conditional Actor and Target Updates (Delayed Policy Update)
    def _update_actor_and_targets(state):
        # Update actor using the new critic params
        new_actor, actor_info = update_actor(state, batch)

        # Soft update target networks
        new_target_critic_params = target_update(new_critic.params, state.target_critic_params, state.config.tau)
        new_target_actor_params = target_update(new_actor.params, state.target_actor_params, state.config.tau)
        return new_actor, new_target_actor_params, new_target_critic_params, actor_info

    def _no_update_actor_and_targets(state):
        # Keep actor and targets the same, return empty actor info
        return state.actor, state.target_actor_params, state.target_critic_params, {'actor_loss': jnp.nan}

    # Use jax.lax.cond for conditional execution on device
    new_actor, new_target_actor_params, new_target_critic_params, actor_info = jax.lax.cond(
        state.step % state.config.policy_delay == 0,
        _update_actor_and_targets,
        _no_update_actor_and_targets,
        state,
    )

    # Create new state with updated values
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        target_actor_params=new_target_actor_params,
        target_critic_params=new_target_critic_params,
        rng=rng,
        step=state.step + 1
    )

    return new_state, {**critic_info, **actor_info}


@jax.jit
def _td3_sample_step(state: TD3State, observation: jnp.ndarray) -> Tuple[TD3State, jnp.ndarray]:
    """Samples an action using the policy, adds exploration noise, and updates RNG state."""
    rng, key_action, key_noise = jax.random.split(state.rng, 3)

    # Get deterministic action from policy
    action = state.actor.apply_fn({'params': state.actor.params}, observation)

    # Add exploration noise
    noise = jax.random.normal(key_noise, action.shape) * state.config.exploration_noise * state.config.max_action
    action = (action + noise).clip(-state.config.max_action, state.config.max_action)

    new_state = state.replace(rng=rng)
    return new_state, action

@jax.jit
def _td3_sample_eval_step(state: TD3State, observation: jnp.ndarray) -> jnp.ndarray:
    """Samples a deterministic action for evaluation (no noise)."""
    action = state.actor.apply_fn({'params': state.actor.params}, observation)
    # No noise added for evaluation, action is already clipped by actor's tanh * max_action
    return action


# --- Factory Function ---
def create_td3_learner(
    seed: int,
    observations: jnp.ndarray, # Sample observation for initialization
    actions: jnp.ndarray,      # Sample action for initialization
    config: TD3Config,
) -> TD3State:
    """Factory function to create TD3 state."""
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    action_dim = actions.shape[-1]

    # Initialize Actor network and state
    actor_def = DeterministicActor(action_dim=action_dim, max_action=config.max_action,
                                   hidden_dims=config.hidden_dims,)
    actor_params = actor_def.init(actor_key, observations)['params']
    actor = train_state.TrainState.create(
        apply_fn=actor_def.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=config.actor_lr)
    )

    # Initialize Critic network and state (DoubleCritic)
    critic_def = DoubleCritic(config.hidden_dims)
    # Use dummy actions with the correct dimension for initialization
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = train_state.TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=config.critic_lr)
    )

    # Create the TD3 state, initializing target networks same as main networks
    return TD3State(
        actor=actor,
        critic=critic,
        target_actor_params=actor_params,
        target_critic_params=critic_params,
        rng=rng,
        step=0,
        config=config
    )


# --- TD3 Agent Class ---
@struct.dataclass
class TD3Agent(RLAgent):
    state: TD3State

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        agent_config: Dict,
        **kwargs # Accept potential extra kwargs
    ) -> "TD3Agent":
        """Creates the TD3Agent with its initial state."""
        # TODO: Add action_scale and action_bias calculations if needed,
        # or ensure max_action handles scaling correctly.
        # For now, assuming max_action is sufficient.
       
        

        initial_state = create_td3_learner(
            seed=seed,
            observations=observations,
            actions=actions, # Pass sample action
            config=TD3Config(**agent_config),
        )
        return cls(state=initial_state)

    def update(self, batch: Batch) -> Tuple["TD3Agent", InfoDict]:
        """Performs a single TD3 training step."""
        new_state, info = _td3_update_step(self.state, batch)
        return TD3Agent(state=new_state), info

    def sample(self, observation: jnp.ndarray) -> Tuple["TD3Agent", jnp.ndarray]:
        """Samples an action stochastically with exploration noise."""
        new_state, action = _td3_sample_step(self.state, observation)
        return TD3Agent(state=new_state), action

    def sample_eval(self, observation: jnp.ndarray) -> Tuple["TD3Agent", jnp.ndarray]:
        """Samples an action deterministically for evaluation."""
        # Evaluation sampling doesn't change the agent's state (no RNG update needed)
        action = _td3_sample_eval_step(self.state, observation)
        return self, action # Return self as state is unchanged 