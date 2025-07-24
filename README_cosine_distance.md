# Cosine Distance Evaluation for TD3-GC

This module provides functionality to evaluate the quality of gradient correction in TD3-GC by measuring cosine distances between different gradient estimates.

## Overview

The `cosine_distance.py` module measures how well the TD3-GC gradient correction approximates the "true" gradient by:

1. **Off-policy gradient**: Computed using current replay buffer data and current critic
2. **Corrected gradient**: TD3-GC corrected version of the off-policy gradient
3. **True gradient**: Computed using fresh policy rollouts and a critic trained from scratch on that data

## Key Metrics

- `cosine_distance/off_policy_vs_true`: How far off-policy gradients are from the true gradient
- `cosine_distance/corrected_vs_true`: How far corrected gradients are from the true gradient  
- `cosine_distance/off_policy_vs_corrected`: Difference between off-policy and corrected gradients
- `gradient_improvement`: Positive values indicate correction is helping (off_policy_vs_true - corrected_vs_true)

## Usage

### Integration into Main Training Loop

```python
from cosine_distance import maybe_evaluate_gradient_quality

# In your training loop (see example_main_with_cosine_eval.py):
if agent_name == "TD3GC":
    gradient_eval_results = maybe_evaluate_gradient_quality(
        agent=agent,
        env=gradient_eval_env,
        replay_buffer=replay_buffer,
        step_num=step_num,
        evaluation_frequency=50_000,  # Every 50K steps
        rollout_steps=100_000,        # 100K rollout steps
        critic_training_steps=50_000, # 50K training steps for fresh critic
        evaluation_batch_size=1000,   # Batch size for gradient comparison
        num_parallel_envs=8           # Use 8 parallel environments for faster rollouts
    )
```

### Manual Evaluation

```python
from cosine_distance import evaluate_gradient_quality

results = evaluate_gradient_quality(
    agent=your_td3gc_agent,
    env=your_environment,
    replay_buffer=your_replay_buffer,
    step_num=current_step,
    rollout_steps=100_000,
    critic_training_steps=50_000,
    evaluation_batch_size=1000,
    num_parallel_envs=8
)
```

## Configuration Parameters

- `rollout_steps` (default: 100,000): Number of steps to rollout with current policy for true gradient
- `critic_training_steps` (default: 50,000): Number of training steps for the fresh critic
- `evaluation_batch_size` (default: 1000): Batch size for gradient comparison
- `evaluation_frequency` (default: 50,000): How often to run evaluation during training
- `num_parallel_envs` (default: 8): Number of parallel environments for vectorized rollout collection

## Expected Results

**Good gradient correction** should show:
- `gradient_improvement > 0`: Corrected gradients are closer to true gradients than off-policy ones
- `cosine_distance/corrected_vs_true < cosine_distance/off_policy_vs_true`: Correction reduces distance to true gradient

**Poor gradient correction** might show:
- `gradient_improvement ≤ 0`: Correction doesn't help or makes things worse
- Very high cosine distances (> 1.5): Gradients are very different from true gradients

## Files

- `cosine_distance.py`: Main module with all evaluation functions
- `example_main_with_cosine_eval.py`: Example of integration into main training loop
- `test_cosine_distance.py`: Test script to validate functionality
- `demo_parallel_rollouts.py`: Performance demonstration of parallel environments
- `README_cosine_distance.md`: This documentation

## Testing

Run the test script to validate the setup:

```bash
python test_cosine_distance.py
```

This will:
1. Test basic cosine distance computation
2. Run a full gradient quality evaluation with reduced parameters
3. Validate all results and provide interpretation

### Performance Demo

To see the speedup benefits of parallel environments:

```bash
python demo_parallel_rollouts.py
```

This demonstrates:
- Performance comparison between single and parallel environment rollouts
- Speedup scaling with different numbers of parallel environments
- Full cosine distance evaluation using parallel environments

## Wandb Logging

The evaluation automatically logs results to Wandb with the following structure:

```
cosine_distance/off_policy_vs_true: float
cosine_distance/corrected_vs_true: float  
cosine_distance/off_policy_vs_corrected: float
gradient_norms/off_policy: float
gradient_norms/corrected: float
gradient_norms/true: float
gradient_improvement: float
step: int
```

## Performance Considerations

- **Evaluation is expensive**: 100K rollouts + 50K critic training steps take significant time
- **Memory usage**: Large rollout datasets (100K steps) require substantial memory
- **Frequency**: Default 50K step frequency balances insight with computational cost
- **Parallel environments**: The module automatically uses [Gymnasium's vectorized environments](https://www.gymlibrary.dev/api/vector/) for much faster rollout collection
- **Speedup**: With 8 parallel environments, rollout collection can be ~8x faster than single environment

### Vectorized Environment Benefits

The system uses `gym.vector.make()` with `asynchronous=True` for parallel rollout collection:
- **Automatic**: Falls back to single environment if vectorization fails
- **Configurable**: Adjust `num_parallel_envs` based on your system (default: 8)
- **Robust**: Handles environment resets and episode boundaries correctly
- **Compatible**: Works with any Gymnasium-registered environment

## Troubleshooting

**"Must provide full state for actor gradient computation"**: Make sure you're passing the full TD3StateGC, not just individual components.

**High memory usage**: Reduce `rollout_steps` or `evaluation_batch_size` for testing.

**Slow evaluation**: Reduce `critic_training_steps` for faster evaluation during development.

**Inconsistent results**: Ensure sufficient training before first evaluation (replay buffer should have diverse data).

**Vectorized environment issues**: If parallel environments fail, the system automatically falls back to single environment collection. Check console output for warnings.

## Integration Notes

The evaluation is designed to:
- Only run for TD3GC agents (automatically detected)
- Use a separate environment instance to avoid interference
- Require sufficient replay buffer data before running
- Log results both to console and Wandb
- Return empty dict when not evaluating (for clean integration)

This provides a robust way to validate that your TD3-GC implementation is working correctly and that the gradient correction is providing meaningful improvements over standard off-policy gradients. 