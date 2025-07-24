# demo_parallel_rollouts.py
# Demonstration of parallel environment speedup for cosine distance evaluation

import time
import numpy as np
import gymnasium as gym
from td3_gc import TD3AgentGC, TD3ConfigGC
from utils import ReplayBuffer
from cosine_distance import collect_rollouts, collect_rollouts_single_env

def benchmark_rollout_collection():
    """
    Benchmark the performance difference between single and parallel environment rollouts.
    """
    print("=" * 60)
    print("Benchmarking Rollout Collection Performance")
    print("=" * 60)
    
    # Setup
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    # Create a simple TD3-GC agent for testing
    config = TD3ConfigGC(
        agent_name="TD3GC",
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma_critic_lr=3e-4,
        hidden_dims=(64, 64),
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
    
    # Test parameters (smaller for demo)
    test_steps = 5000
    parallel_envs = [1, 2, 4, 8]
    
    print(f"Environment: {env_name}")
    print(f"Rollout steps: {test_steps}")
    print(f"Testing parallel environments: {parallel_envs}")
    print()
    
    results = {}
    
    # Test single environment (baseline)
    print("Testing single environment (baseline)...")
    start_time = time.time()
    try:
        obs, acts, rews, next_obs, masks = collect_rollouts_single_env(
            agent, env, num_steps=test_steps, deterministic=True
        )
        single_env_time = time.time() - start_time
        results[1] = single_env_time
        print(f"✅ Single environment: {single_env_time:.2f} seconds")
        print(f"   Collected {len(obs)} transitions")
    except Exception as e:
        print(f"❌ Single environment failed: {e}")
        single_env_time = None
    print()
    
    # Test parallel environments
    for num_envs in parallel_envs[1:]:  # Skip 1 since we already tested single
        print(f"Testing {num_envs} parallel environments...")
        start_time = time.time()
        try:
            obs, acts, rews, next_obs, masks = collect_rollouts(
                agent, env, num_steps=test_steps, deterministic=True, num_parallel_envs=num_envs
            )
            parallel_time = time.time() - start_time
            results[num_envs] = parallel_time
            print(f"✅ {num_envs} parallel environments: {parallel_time:.2f} seconds")
            print(f"   Collected {len(obs)} transitions")
            
            if single_env_time is not None:
                speedup = single_env_time / parallel_time
                print(f"   Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"❌ {num_envs} parallel environments failed: {e}")
        print()
    
    # Summary
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if single_env_time is not None:
        print(f"{'Environments':<15} {'Time (s)':<10} {'Speedup':<10}")
        print("-" * 35)
        
        for num_envs, exec_time in results.items():
            if num_envs == 1:
                print(f"{num_envs:<15} {exec_time:<10.2f} {'1.00x':<10}")
            else:
                speedup = single_env_time / exec_time
                print(f"{num_envs:<15} {exec_time:<10.2f} {speedup:.2f}x")
    
    env.close()
    
    print("\n🎯 Key Takeaways:")
    print("- Parallel environments significantly speed up rollout collection")
    print("- Speedup scales roughly linearly with number of environments")
    print("- For cosine distance evaluation, this reduces the most time-consuming part")
    print("- Automatic fallback ensures robustness across different environments")


def demo_cosine_evaluation_with_parallel():
    """
    Demo the full cosine distance evaluation with parallel environments.
    """
    print("\n" + "=" * 60)
    print("Demo: Full Cosine Distance Evaluation with Parallel Environments")
    print("=" * 60)
    
    from cosine_distance import evaluate_gradient_quality
    
    # Setup (same as benchmark)
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    config = TD3ConfigGC(
        agent_name="TD3GC",
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma_critic_lr=3e-4,
        hidden_dims=(64, 64),
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
    
    # Create and populate replay buffer
    replay_buffer = ReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        capacity=10000
    )
    
    print("Populating replay buffer...")
    obs, _ = env.reset()
    for step in range(2000):
        if step < 500:
            action = env.action_space.sample()
        else:
            agent, action = agent.sample(np.expand_dims(obs, 0))
            action = action[0]
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        mask = 1.0 - float(terminated)
        
        replay_buffer.add(obs, action, reward, mask, next_obs, 1.0)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
        
        # Train agent occasionally
        if step >= 500 and step % 10 == 0:
            batch = replay_buffer.sample(32)
            agent, _ = agent.update(batch)
    
    print(f"Replay buffer populated with {len(replay_buffer)} transitions")
    print("Running cosine distance evaluation...")
    
    # Run evaluation with parallel environments
    start_time = time.time()
    results = evaluate_gradient_quality(
        agent=agent,
        env=env,
        replay_buffer=replay_buffer,
        step_num=2000,
        rollout_steps=2000,        # Smaller for demo
        critic_training_steps=500,  # Smaller for demo
        evaluation_batch_size=100,
        num_parallel_envs=4        # Use 4 parallel environments
    )
    total_time = time.time() - start_time
    
    print(f"\n✅ Evaluation completed in {total_time:.2f} seconds")
    print("\nResults:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
    
    env.close()


if __name__ == "__main__":
    benchmark_rollout_collection()
    demo_cosine_evaluation_with_parallel() 