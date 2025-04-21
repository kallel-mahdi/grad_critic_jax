# In a new file, e.g., 'base_agent.py' or within your main script
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import flax
import jax.numpy as jnp
from flax import struct
from jax import random

# Common Type Definitions
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Batch = flax.struct.dataclass
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

class RLAgentConfig(struct.PyTreeNode):
    agent_name: str = struct.field(pytree_node=False)

class RLAgentState(struct.PyTreeNode):
    rng: PRNGKey
    step: int
    # Add other common state elements if any (e.g., config)


@struct.dataclass
class RLAgent(ABC):
    state: RLAgentState

    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> RLAgentState:
        """Creates the initial agent state."""
        pass

    @abstractmethod
    def update(self, state: RLAgentState, batch: Batch) -> Tuple[RLAgentState, InfoDict]:
        """Performs a single training step."""
        pass

    @abstractmethod
    def sample(self, state: RLAgentState, observation: jnp.ndarray) -> Tuple[RLAgentState, jnp.ndarray]:
        """Samples an action for a given observation."""
        pass
    
    @abstractmethod
    def sample_eval(self, state: RLAgentState, observation: jnp.ndarray) -> Tuple[RLAgentState, jnp.ndarray]:
        """Samples an action for evaluation (often deterministic)."""
        pass
