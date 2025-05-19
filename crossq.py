from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
from flax.training import train_state
from flax import struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
import distrax

# Import common components from utils
from utils import Batch, PRNGKey, Params, InfoDict, BatchRenorm, SimbaResidualBlock
# Import base agent class
from base_agent import RLAgent, RLAgentState, RLAgentConfig
# Import Temperature from networks
from networks import Temperature

@struct.dataclass
class CrossQConfig(RLAgentConfig):
    actor_lr: float
    critic_lr: float
    temp_lr: float
    hidden_dims: Sequence[int]
    discount: float
    target_entropy: float
    backup_entropy: bool
    init_temperature: float
    policy_log_std_min: float
    policy_log_std_max: float
    policy_final_fc_init_scale: float
    target_entropy_multiplier: float
    max_action: float
    # BatchRenorm specific params
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    use_batch_norm: bool = True
    n_critics: int = 2
    use_simba_layers: bool = False
    scale_factor: int = 4
    policy_delay: int = 1

# New state class for batch stats tracking
class BatchNormTrainState(train_state.TrainState):
    batch_stats: FrozenDict

@struct.dataclass
class CrossQState(RLAgentState):
    rng: PRNGKey
    step: int
    actor: BatchNormTrainState
    critic: BatchNormTrainState
    temp: train_state.TrainState
    config: CrossQConfig  # Store configuration in the state

# --- Actor Network with BatchRenorm ---
class StochasticActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    final_fc_init_scale: float = 1.0
    use_simba_layers: bool = False
    scale_factor: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, temperature: float = 1.0, train: bool = False) -> Any:
        # Flatten input if needed - assuming x has shape [batch_size, *observation_dims]
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        # Apply BatchRenorm initially if used
        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        
        # Process through hidden layers with appropriate normalization
        if self.use_simba_layers:
            # SimbaResidualBlock approach
            for n_units in self.hidden_dims:
                x = SimbaResidualBlock(
                    n_units,
                    nn.relu,
                    self.scale_factor,
                    lambda: BatchRenorm(
                        use_running_average=not train,
                        momentum=self.batch_norm_momentum,
                        warmup_steps=self.renorm_warmup_steps,
                    ),
                )(x)
            
            # Final normalization before output
            if self.use_batch_norm:
                x = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)
        else:
            # Standard MLP with BatchRenorm between layers
            for i, n_units in enumerate(self.hidden_dims):
                x = nn.Dense(n_units, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
                x = nn.relu(x)
                if self.use_batch_norm:
                    x = BatchRenorm(
                        use_running_average=not train,
                        momentum=self.batch_norm_momentum,
                        warmup_steps=self.renorm_warmup_steps,
                    )(x)
        
        # Policy head outputs mean and log_std
        mean = nn.Dense(
            self.action_dim, 
            kernel_init=nn.initializers.orthogonal(self.final_fc_init_scale)
        )(x)
        
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(self.final_fc_init_scale)
        )(x)
        
        # Clip log_std to specified range
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        # Create distribution
        base_dist = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(log_std) * temperature
        )
        
        # Apply tanh transformation
        return distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), ndims=1))


# --- Critic Network with BatchRenorm ---
class Critic(nn.Module):
    hidden_dims: Sequence[int]
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    use_simba_layers: bool = False
    scale_factor: int = 4
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # Flatten observations if needed
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)
        
        # Concatenate observations and actions
        x = jnp.concatenate([observations, actions], -1)
        
        # Apply BatchRenorm initially if used
        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        
        # Process through hidden layers with appropriate normalization
        if self.use_simba_layers:
            # SimbaResidualBlock approach
            x = nn.Dense(self.hidden_dims[0])(x)
            
            for n_units in self.hidden_dims:
                x = SimbaResidualBlock(
                    n_units,
                    nn.relu,
                    self.scale_factor,
                    lambda: BatchRenorm(
                        use_running_average=not train,
                        momentum=self.batch_norm_momentum,
                        warmup_steps=self.renorm_warmup_steps,
                    ),
                )(x)
            
            # Final normalization before output
            if self.use_batch_norm:
                x = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)
        else:
            # Standard MLP with BatchRenorm between layers
            for i, n_units in enumerate(self.hidden_dims):
                x = nn.Dense(n_units, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
                x = nn.relu(x)
                if self.use_batch_norm:
                    x = BatchRenorm(
                        use_running_average=not train,
                        momentum=self.batch_norm_momentum,
                        warmup_steps=self.renorm_warmup_steps,
                    )(x)
        
        # Output layer
        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(x, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    num_qs: int = 2
    use_simba_layers: bool = False
    scale_factor: int = 4

    @nn.compact
    def __call__(self, states, actions, train: bool = False):
        # Creates multiple Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(
            Critic,
            variable_axes={'params': 0, 'batch_stats': 0},  # Map over parameters and batch stats
            split_rngs={'params': True, 'batch_stats': True},  # Split RNGs for each critic
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs  # Number of critics to create
        )
        
        qs = VmapCritic(
            self.hidden_dims,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            renorm_warmup_steps=self.renorm_warmup_steps,
            use_simba_layers=self.use_simba_layers,
            scale_factor=self.scale_factor
        )(states, actions, train)
        
        return qs


# --- Update Functions ---

def update_temperature(state: CrossQState, entropy: float) -> Tuple[train_state.TrainState, InfoDict]:
    # Defines the loss function for temperature optimization.
    def temperature_loss_fn(temp_params):
        temperature = state.temp.apply_fn({'params': temp_params})
        # Loss aims to match current entropy to target entropy.
        temp_loss = temperature * (entropy - state.config.target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    # Compute gradients and update temperature using TrainState's apply_gradients
    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(state.temp.params)
    new_temp = state.temp.apply_gradients(grads=grads)

    return new_temp, info


def update_actor(key_actor: PRNGKey, state: CrossQState, batch: Batch) -> Tuple[BatchNormTrainState, InfoDict]:
    # Get temperature value (scalar)
    temperature = state.temp.apply_fn({'params': state.temp.params})

    def actor_loss_fn(actor_params):
        # Apply actor model to get action distribution using actor's apply_fn
        variables = {'params': actor_params, 'batch_stats': state.actor.batch_stats}
        dist = state.actor.apply_fn(variables, batch.observations, train=True, mutable=['batch_stats'])
        
        # Handle the returned mutables
        model_output, new_model_state = dist
        actions, log_probs = model_output.sample_and_log_prob(seed=key_actor)

        # Evaluate actions using the critic network's apply_fn
        critic_variables = {'params': state.critic.params, 'batch_stats': state.critic.batch_stats}
        q_values, _ = state.critic.apply_fn(critic_variables, batch.observations, actions, train=False, mutable=False)
        # Take min across critics
        q = jnp.min(q_values, axis=0)

        # Actor loss: encourages actions with high Q-values and high entropy.
        actor_loss = (log_probs * temperature - q).mean()

        # Return loss, new batch_stats, and auxiliary info (entropy).
        return actor_loss, (new_model_state, {'actor_loss': actor_loss, 'entropy': -log_probs.mean()})

    # Compute gradients and update actor model using TrainState's apply_gradients
    (actor_loss, (new_model_state, info)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)
    
    # Update actor
    new_actor = state.actor.apply_gradients(grads=grads)
    # Update batch_stats with new values
    new_actor = new_actor.replace(batch_stats=new_model_state['batch_stats'])

    return new_actor, info


def update_critic(key_critic: PRNGKey, state: CrossQState, batch: Batch) -> Tuple[BatchNormTrainState, InfoDict]:
    # Key for sampling the next actions
    key_next_action = key_critic
    
    # Get temperature value (scalar)
    temperature = state.temp.apply_fn({'params': state.temp.params})
    
    # Get next actions and log probabilities from the actor for the *next* observations.
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    actor_output = state.actor.apply_fn(variables, batch.next_observations, train=False, mutable=False)
    next_actions, next_log_probs = actor_output.sample_and_log_prob(seed=key_next_action)
    
    # The key innovation: process both current and next observation-action pairs through the critic
    # This ensures BatchRenorm statistics are calculated on the joint distribution
    joint_observations = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
    joint_actions = jnp.concatenate([batch.actions, next_actions], axis=0)
    
    def critic_loss_fn(critic_params):
        # Joint forward pass through critic with both current and next pairs
        variables = {'params': critic_params, 'batch_stats': state.critic.batch_stats}
        joint_output = state.critic.apply_fn(variables, joint_observations, joint_actions, train=True, mutable=['batch_stats'])
        
        # Handle the returned values and mutables
        joint_q_values, new_model_state = joint_output
        
        # Split back into current and next Q-values
        batch_size = batch.observations.shape[0]
        current_q_values, next_q_values = joint_q_values[:, :batch_size], joint_q_values[:, batch_size:]
        
        # Compute target Q-values (minimum across critics)
        min_next_q = jnp.min(next_q_values, axis=0)
        
        # # Apply entropy adjustment if specified
        # if state.config.backup_entropy:
        min_next_q -= temperature * next_log_probs
        
        # Bellman target calculation
        target_q = batch.rewards + state.config.discount * batch.masks * min_next_q
        target_q = jax.lax.stop_gradient(target_q)
        
        # Compute losses for all critics
        critic_losses = jnp.mean((current_q_values - target_q) ** 2, axis=1)
        critic_loss = jnp.sum(critic_losses)
        
        # Return loss, new batch_stats, and auxiliary info
        return critic_loss, (new_model_state, {
            'critic_loss': critic_loss,
            'q_values': jnp.mean(current_q_values, axis=1)
        })
    
    # Compute gradients and update critic
    (critic_loss, (new_model_state, info)), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic.params)
    
    # Update critic
    new_critic = state.critic.apply_gradients(grads=grads)
    # Update batch_stats with new values
    new_critic = new_critic.replace(batch_stats=new_model_state['batch_stats'])
    
    return new_critic, info


# --- Update Steps (External JIT functions) ---

@jax.jit
def _crossq_update_step(state: CrossQState, batch: Batch) -> Tuple[CrossQState, InfoDict]:
    """Single update step for all components (external JIT)."""
    rng, key_critic, key_actor = jax.random.split(state.rng, 3)

    # Update critic
    new_critic, critic_info = update_critic(key_critic, state, batch)

    # Update actor with policy delay
    should_update_actor = (state.step % state.config.policy_delay == 0)
    
    def _update_actor():
        # Update actor using updated critic
        temp_state = state.replace(critic=new_critic)
        new_actor, actor_info = update_actor(key_actor, temp_state, batch)
        
        # Update temperature using updated actor entropy
        temp_state_for_temp = temp_state.replace(actor=new_actor)
        new_temp, alpha_info = update_temperature(temp_state_for_temp, actor_info['entropy'])
        
        return new_actor, new_temp, {**actor_info, **alpha_info}
    
    def _no_update_actor():
        return state.actor, state.temp, {'actor_loss': jnp.array(0.0), 'entropy': jnp.array(0.0), 'temperature': jnp.array(0.0), 'temp_loss': jnp.array(0.0)}
    
    new_actor, new_temp, actor_temp_info = jax.lax.cond(
        should_update_actor,
        lambda: _update_actor(),
        lambda: _no_update_actor()
    )

    # Create new state with updated values
    new_state = state.replace(
        actor=new_actor,
        critic=new_critic,
        temp=new_temp,
        rng=rng,
        step=state.step + 1
    )

    return new_state, {**critic_info, **actor_temp_info}


@jax.jit
def _crossq_sample_step(state: CrossQState, observation: jnp.ndarray) -> Tuple[CrossQState, jnp.ndarray]:
    """Samples an action using the policy and updates RNG state (external JIT)."""
    rng, key = jax.random.split(state.rng)
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    dist = state.actor.apply_fn(variables, observation, train=False, mutable=False)
    action = dist.sample(seed=key)
    new_state = state.replace(rng=rng)
    return new_state, action

@jax.jit
def _crossq_sample_eval_step(state: CrossQState, observation: jnp.ndarray) -> jnp.ndarray:
    """Samples a deterministic action for evaluation (external JIT)."""
    variables = {'params': state.actor.params, 'batch_stats': state.actor.batch_stats}
    dist = state.actor.apply_fn(variables, observation, temperature=0.0, train=False, mutable=False)
    action = dist.sample(seed=state.rng)

    return action


# --- Factory Function ---
def create_crossq_learner(
    seed: int,
    observations: jnp.ndarray,  # Sample observation for initialization
    actions: jnp.ndarray,       # Sample action for initialization
    config: CrossQConfig,
) -> CrossQState:
    """Factory function to create CrossQ state."""
    
    # Initialize PRNG keys
    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    
    # Initialize Actor network and state
    actor_def = StochasticActor(
        config.hidden_dims,
        actions.shape[-1],
        max_action=config.max_action,
        log_std_min=config.policy_log_std_min,
        log_std_max=config.policy_log_std_max,
        final_fc_init_scale=config.policy_final_fc_init_scale,
        use_batch_norm=config.use_batch_norm,
        batch_norm_momentum=config.batch_norm_momentum,
        renorm_warmup_steps=config.renorm_warmup_steps,
        use_simba_layers=config.use_simba_layers,
        scale_factor=config.scale_factor
    )
    
    # Initialize actor
    actor_variables = actor_def.init(actor_key, observations, train=False)
    
    # Create BatchNormTrainState for actor
    actor = BatchNormTrainState.create(
        apply_fn=actor_def.apply,
        params=actor_variables['params'],
        batch_stats=actor_variables.get('batch_stats', FrozenDict({})),
        tx=optax.adam(learning_rate=config.actor_lr)
    )
    
    # Initialize Critic network and state
    critic_def = DoubleCritic(
        config.hidden_dims,
        use_batch_norm=config.use_batch_norm,
        batch_norm_momentum=config.batch_norm_momentum,
        renorm_warmup_steps=config.renorm_warmup_steps,
        num_qs=config.n_critics,
        use_simba_layers=config.use_simba_layers,
        scale_factor=config.scale_factor
    )
    
    # Initialize critic
    critic_variables = critic_def.init(critic_key, observations, actions, train=False)
    
    # Create BatchNormTrainState for critic
    critic = BatchNormTrainState.create(
        apply_fn=critic_def.apply,
        params=critic_variables['params'],
        batch_stats=critic_variables.get('batch_stats', FrozenDict({})),
        tx=optax.adam(learning_rate=config.critic_lr)
    )
    
    # Initialize Temperature parameter and state
    temp_def = Temperature(config.init_temperature)
    temp_params = temp_def.init(temp_key)['params']
    temp = train_state.TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(learning_rate=config.temp_lr)
    )
    
    # Create the CrossQ state
    return CrossQState(
        actor=actor,
        critic=critic,
        temp=temp,
        rng=rng,
        step=0,
        config=config
    )


# --- CrossQ Agent Class ---
@struct.dataclass
class CrossQAgent(RLAgent):
    state: CrossQState

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        agent_config: Dict,
        **kwargs  # Accept potential extra kwargs from base class
    ) -> "CrossQAgent":
        """Creates the CrossQAgent with its initial state."""
        # Set target entropy if not already specified
        agent_config["target_entropy"] = -actions.shape[-1] * agent_config["target_entropy_multiplier"]
        
        # Create initial state
        initial_state = create_crossq_learner(
            seed=seed,
            observations=observations,
            actions=actions,
            config=CrossQConfig(**agent_config),
        )
        
        return cls(state=initial_state)

    def update(self, batch: Batch) -> Tuple["CrossQAgent", InfoDict]:
        """Performs a single CrossQ training step using the external JIT function."""
        new_state, info = _crossq_update_step(self.state, batch)
        # Return a new agent instance wrapping the new state
        return CrossQAgent(state=new_state), info

    def sample(self, observation: jnp.ndarray) -> Tuple["CrossQAgent", jnp.ndarray]:
        """Samples an action stochastically using the external JIT function."""
        new_state, action = _crossq_sample_step(self.state, observation)
        # Return a new agent instance wrapping the new state (with updated RNG)
        return CrossQAgent(state=new_state), action

    def sample_eval(self, observation: jnp.ndarray) -> Tuple["CrossQAgent", jnp.ndarray]:
        """Samples an action deterministically using the external JIT function."""
        action = _crossq_sample_eval_step(self.state, observation)
        # Evaluation sampling doesn't change the agent's state (no RNG update)
        return self, action