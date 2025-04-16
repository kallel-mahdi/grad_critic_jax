import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from sac import create_sac_learner, SACConfig, _update_step as sac_update_step  # Import from your sac.py
from sac_gc import create_sac_learner as create_sac_gc_learner, SACConfig as SACGCConfig, _update_step as sac_gc_update_step  # Import from your sac_gc.py
from utils import ReplayBuffer, evaluate, sample_actions
import wandb

# Set Weights & Biases API key
os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
#s.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # --- Setup ---
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # Initialize Weights & Biases
    if cfg.logging.use_wandb:
        # Construct meaningful run name
        run_name = f"{cfg.env.name}_{cfg.training.max_steps}steps"
        if cfg.algorithm.gamma_correction: run_name += "_gc"
        run_name += f"_seed{cfg.seed}"

        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True), # Log resolved config
            name=run_name,
            sync_tensorboard=False, # Disabled TensorBoard sync
            monitor_gym=True,      # Automatically logs videos of env
            save_code=True,        # Saves the main script to W&B
            dir=output_dir         # Save W&B logs in Hydra's output dir
            )

    # Create environment
    env = gym.make(cfg.env.name)
    eval_env = gym.make(cfg.env.name)  # Separate env for evaluation

    # Set random seeds
    np.random.seed(cfg.seed)
    # Consider setting JAX seed here as well if not done in create_learner

    # Get observation and action space info
    obs_space = env.observation_space
    act_space = env.action_space
    sample_obs = np.zeros((1, obs_space.shape[0]), dtype=np.float32)
    sample_act = np.zeros((1, act_space.shape[0]), dtype=np.float32)
    target_entropy = -act_space.shape[0] * cfg.algorithm.target_entropy_multiplier # Calculate target entropy

    # Create SAC config (adapt based on SACConfig definition)
    # Assuming SACConfig can accept these named arguments directly from DictConfig
    # Or convert cfg.algorithm to a compatible format if needed
    agent_config = SACConfig( # or SACGCConfig if they are different
        actor_lr=cfg.algorithm.actor_lr,
        critic_lr=cfg.algorithm.critic_lr,
        temp_lr=cfg.algorithm.temp_lr,
        hidden_dims=tuple(cfg.algorithm.hidden_dims), # Ensure it's a tuple
        discount=cfg.algorithm.discount,
        tau=cfg.algorithm.tau,
        target_update_period=cfg.algorithm.target_update_period,
        target_entropy=target_entropy,
        backup_entropy=cfg.algorithm.backup_entropy,
        init_temperature=cfg.algorithm.init_temperature
    )

    # Initialize SAC or SAC-GC agent based on gamma_correction flag
    if cfg.algorithm.gamma_correction:
        # Ensure create_sac_gc_learner and SACGCConfig use the correct parameters
        print("Using SAC with Gradient Critic (SAC-GC)")
        state = create_sac_gc_learner(
            seed=cfg.seed,
            observations=sample_obs,
            actions=sample_act,
            config=agent_config # Pass the created agent_config
        )
    else:
        print("Using standard SAC")
        state = create_sac_learner(
            seed=cfg.seed,
            observations=sample_obs,
            actions=sample_act,
            config=agent_config # Pass the created agent_config
        )

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        capacity=cfg.training.replay_buffer_capacity
    )

    # Training loop
    observation, _ = env.reset(seed=cfg.seed)
    for step_num in tqdm(range(1, cfg.training.max_steps + 1), desc=f"Training {cfg.env.name}"):
        if step_num < cfg.training.start_steps:
            # Sample random actions before training starts
            action = env.action_space.sample()
        else:
            # Sample actions from the policy
            state, action = sample_actions(state, observation) # Assuming sample_actions takes state and obs

        # Execute action in environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        mask = 1.0 - float(terminated)  # Mask is 1 if not terminated (for critic target)

        # Store transition in replay buffer
        replay_buffer.add(observation, action, reward, mask, next_observation)

        # Update observation
        observation = next_observation

        # Reset environment if episode ends
        if done:
            observation, _ = env.reset()
            if cfg.logging.use_wandb and 'episode' in info:
                 wandb.log({'train/episode_return': info['episode']['r'][0],
                            'train/episode_length': info['episode']['l'][0],
                            'step': step_num}, commit=False)

        # Perform agent update if enough steps have passed
        if step_num >= cfg.training.start_steps:
            batch = replay_buffer.sample(cfg.training.batch_size)
            if cfg.algorithm.gamma_correction:
                state, update_info = sac_gc_update_step(state, batch)  # Use SAC-GC update
            else:
                state, update_info = sac_update_step(state, batch)  # Use SAC update

            if cfg.logging.use_wandb and update_info:
                wandb.log({f'train/{k}': v for k, v in update_info.items()}, commit=False)

        # Evaluate agent periodically
        if step_num % cfg.training.eval_freq == 0:
            eval_returns = []
            for _ in range(cfg.training.eval_episodes):
                 eval_state, avg_reward = evaluate(state, eval_env, 1) # Evaluate one episode at a time
                 eval_returns.append(avg_reward)
                 state = state.replace(rng=eval_state.rng) # Update RNG state

            avg_eval_reward = np.mean(eval_returns)
            # Log evaluation results to wandb
            log_data = {"eval/avg_reward": avg_eval_reward, "step": step_num}
            if cfg.logging.use_wandb:
                wandb.log(log_data, commit=True) # Commit updates here
            else: # Commit previous train updates if not using wandb eval commit
                 if step_num >= cfg.training.start_steps and update_info and cfg.logging.use_wandb:
                     wandb.log({},commit=True)

            print(f"---------------------------------------")
            print(f"Step: {step_num}, Evaluation Avg Reward: {avg_eval_reward:.2f}")
            print(f"---------------------------------------")

    env.close()
    eval_env.close()
    if cfg.logging.use_wandb:
        wandb.finish()  # Close wandb
    print("Training finished.")

if __name__ == "__main__":
    main()