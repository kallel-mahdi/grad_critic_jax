# test_cosine_distance.py
# Test script to demonstrate cosine distance evaluation functionality

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from td3_gc import TD3AgentGC, TD3ConfigGC
from utils import ReplayBuffer
from cosine_distance import evaluate_gradient_quality

def test_cosine_distance_evaluation():
    """
    Test the cosine distance evaluation functionality with a simple setup.
    """
    print("Testing Cosine Distance Evaluation for TD3-GC")
    print("=" * 50)
    
    # Create test environment
    env_name = "Pendulum-v1"  # Simple continuous control environment
    env = gym.make(env_name)
    
    # Set up basic configuration
    config = TD3ConfigGC(
        agent_name="TD3GC",
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma_critic_lr=3e-4,
        hidden_dims=(64, 64),  # Smaller networks for faster testing
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        exploration_noise=0.1,
        max_action=1.0,
        target_gamma_critic_update_period=2,
        use_layer_norm=False
    )
    
    # Initialize agent
    obs_space = env.observation_space
    act_space = env.action_space
    sample_obs = np.zeros((1, obs_space.shape[0]), dtype=np.float32)
    sample_act = np.zeros((1, act_space.shape[0]), dtype=np.float32)
    
    agent = TD3AgentGC.create(
        seed=42,
        observations=sample_obs,
        actions=sample_act,
        agent_config=config.__dict__
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        capacity=50000
    )
    
    # Collect some initial data
    print("Collecting initial data for replay buffer...")
    obs, _ = env.reset()
    for step in range(5000):  # Reduced for testing
        if step < 1000:
            action = env.action_space.sample()
        else:
            agent, action = agent.sample(obs)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        mask = 1.0 - float(terminated)
        
        replay_buffer.add(obs, action, reward, mask, next_obs, 1.0)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
        
        # Train agent periodically
        if step >= 1000 and step % 4 == 0:
            for _ in range(2):
                batch = replay_buffer.sample(64)
                agent, _ = agent.update(batch)
    
    print(f"Collected {len(replay_buffer)} transitions")
    print(f"Agent has been trained for {agent.state.step} steps")
    
    # Now test the gradient evaluation with reduced parameters for speed
    print("\nRunning gradient quality evaluation...")
    try:
        results = evaluate_gradient_quality(
            agent=agent,
            env=env,
            replay_buffer=replay_buffer,
            step_num=5000,
            rollout_steps=1000,        # Much smaller for testing
            critic_training_steps=500,  # Much smaller for testing
            evaluation_batch_size=100,  # Smaller batch size
            num_parallel_envs=4        # Use 4 parallel environments for testing
        )
        
        print("\nGradient Quality Evaluation Results:")
        print("-" * 40)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        
        # Validate results
        assert 0.0 <= results['cosine_distance/off_policy_vs_true'] <= 2.0
        assert 0.0 <= results['cosine_distance/corrected_vs_true'] <= 2.0
        assert 0.0 <= results['cosine_distance/off_policy_vs_corrected'] <= 2.0
        assert results['gradient_norms/off_policy'] >= 0.0
        assert results['gradient_norms/corrected'] >= 0.0
        assert results['gradient_norms/true'] >= 0.0
        
        print("\n✅ All validation checks passed!")
        
        # Interpret results
        print("\nResult Interpretation:")
        print("-" * 40)
        if results['gradient_improvement'] > 0:
            print(f"✅ Gradient correction is helping! Improvement: {results['gradient_improvement']:.6f}")
        else:
            print(f"❌ Gradient correction may not be helping. Change: {results['gradient_improvement']:.6f}")
        
        print(f"Off-policy gradient is {results['cosine_distance/off_policy_vs_true']:.6f} cosine distance from true gradient")
        print(f"Corrected gradient is {results['cosine_distance/corrected_vs_true']:.6f} cosine distance from true gradient")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise
    
    finally:
        env.close()
    
    print("\n🎉 Test completed successfully!")


def test_cosine_distance_computation():
    """Test the basic cosine distance computation function."""
    print("\nTesting basic cosine distance computation...")
    
    from cosine_distance import compute_cosine_distance
    
    # Test identical vectors
    vec1 = jnp.array([1.0, 2.0, 3.0])
    vec2 = jnp.array([1.0, 2.0, 3.0])
    dist = compute_cosine_distance(vec1, vec2)
    print(f"Distance between identical vectors: {dist:.6f} (should be ~0.0)")
    assert abs(dist) < 1e-6
    
    # Test orthogonal vectors
    vec1 = jnp.array([1.0, 0.0])
    vec2 = jnp.array([0.0, 1.0])
    dist = compute_cosine_distance(vec1, vec2)
    print(f"Distance between orthogonal vectors: {dist:.6f} (should be ~1.0)")
    assert abs(dist - 1.0) < 1e-6
    
    # Test opposite vectors
    vec1 = jnp.array([1.0, 0.0])
    vec2 = jnp.array([-1.0, 0.0])
    dist = compute_cosine_distance(vec1, vec2)
    print(f"Distance between opposite vectors: {dist:.6f} (should be ~2.0)")
    assert abs(dist - 2.0) < 1e-6
    
    print("✅ Basic cosine distance tests passed!")


if __name__ == "__main__":
    # Run tests
    test_cosine_distance_computation()
    test_cosine_distance_evaluation() 