from typing import Callable, Sequence, Optional

import flax.linen as nn
import jax.numpy as jnp
import distrax
# --- Common Network Utilities ---



def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False # Changed default to False for typical MLP usage
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            #x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.Dense(size,kernel_init=default_init())(x)
            #x = nn.Dense(size)(x)
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                
                x = self.activations(x)
              
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                      
        return x
    




# --- Critic Network ---

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_layer_norm: bool = False
    
    @nn.compact
    def setup(self):
        # Main network excluding final layer
        self.feature_net = MLP(self.hidden_dims, activate_final=True,use_layer_norm=self.use_layer_norm)
        # Gamma parameter output head
        self.critic_head = nn.Dense(1, kernel_init=default_init())

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        features = self.feature_net(inputs)
        critic = self.critic_head(features)
        return jnp.squeeze(critic, -1)

class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, states, actions):
        # Creates two Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0}, # Map over parameters for each critic
                             split_rngs={'params': True}, # Use different RNGs for parameter initialization
                             in_axes=None,               # Inputs (states, actions) are shared
                             out_axes=0,               # Stack outputs along the first axis
                             axis_size=self.num_qs)      # Number of critics to create
        qs = VmapCritic(self.hidden_dims,
                        use_layer_norm=self.use_layer_norm,
                        activations=self.activations)(states, actions)
        return qs[0], qs[1] # Return the two Q-values separately


# --- Actor Network ---
class DeterministicActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float # Needed for TD3's action output clipping/scaling if not handled by scale/bias
    dropout_rate: float = None
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, 
                 observations: jnp.ndarray,
                 training: bool = False ):
        
    
        x = MLP((*self.hidden_dims,self.action_dim),
                use_layer_norm=self.use_layer_norm,
                activations=nn.relu,
                activate_final=False,
                dropout_rate=self.dropout_rate)(observations,
                                               training=training)
        action = nn.tanh(x)
        return action



class StochasticActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float
    log_std_min: float  
    log_std_max: float    
    dropout_rate: float = None
    tanh_squash_distribution: bool = True
    
    

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float=1.0,
                 training: bool = False):
        
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         )(outputs)
     
    
        log_stds = nn.Dense(self.action_dim,
                            )(outputs)


        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # TFP uses scale_diag for MultivariateNormalDiag
        base_dist = distrax.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) * temperature)

        if self.tanh_squash_distribution:
           
            return distrax.Transformed(
            base_dist, distrax.Block(distrax.Tanh(), ndims=1))
        else:
            # Returns the raw Normal distribution without tanh squashing.
            return base_dist
        

class Temperature(nn.Module):
    initial_temperature: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        # Parameter for the log of temperature, ensures temperature > 0.
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


