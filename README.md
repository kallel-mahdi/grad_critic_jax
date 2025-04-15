# Gradient Critic SAC Implementation in JAX

This repository contains an implementation of Soft Actor-Critic (SAC) with a gradient critic in JAX. The gradient critic is designed to improve policy optimization by learning the gradients of the actor's loss function directly.

## Features
- SAC implementation in JAX
- Gradient critic for improved policy optimization
- Support for continuous control environments
- Weights & Biases integration for experiment tracking

## Requirements
- JAX
- Flax
- Gymnasium
- Optax
- Distrax
- Wandb

## Usage
```python
python sac_jax2.py --env_name Walker2d-v4 --gamma_correction True
```

## Arguments
- `--env_name`: Gym environment name (default: 'Walker2d-v4')
- `--gamma_correction`: Enable gamma correction for actor updates
- `--use_wandb`: Enable Weights & Biases logging
- See script for full list of arguments

## License
MIT