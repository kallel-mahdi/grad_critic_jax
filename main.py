import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from sac import SACAgent
from sac_gc import SACAgentGC
from td3 import TD3Agent
from td3_gc import TD3AgentGC
from crossq import CrossQAgent
from utils import ReplayBuffer, evaluate

import wandb


os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
# os.environ["HTTPS_PROXY"] = "http://proxy:80"
# os.environ["WANDB__SERVICE_WAIT"] = "300"
# os.environ["WANDB_INIT_TIMEOUT"] = "600"


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    # --- Setup ---
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")

    # Initialize Weights & Biases
    if cfg.logging.use_wandb:
        # Construct meaningful run name
        run_name = f"{cfg.algorithm.agent_name}_{cfg.environment.name}_{cfg.seed}"

        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True), # Log resolved config
            name=run_name,
            sync_tensorboard=False, # Disabled TensorBoard sync
            monitor_gym=False,      # Automatically logs videos of env
            settings=wandb.Settings(
            start_method="thread",
            _disable_stats=False,
            init_timeout=1800,
            ),
            save_code=True,        # Saves the main script to W&B
            dir=output_dir         # Save W&B logs in Hydra's output dir
            )

    # Create environment
    env = gym.make(cfg.environment.name)
    eval_env = gym.make(cfg.environment.name)  # Separate env for evaluation

    # Set random seeds
    np.random.seed(cfg.seed)
    # Consider setting JAX seed here as well if not done in create_learner

    # Get observation and action space info
    obs_space = env.observation_space
    act_space = env.action_space
    sample_obs = np.zeros((1, obs_space.shape[0]), dtype=np.float32)
    sample_act = np.zeros((1, act_space.shape[0]), dtype=np.float32)
    
    agent_config = OmegaConf.to_container(cfg.algorithm, resolve=True)

    
    # --- Agent Initialization (Dynamic) ---
    agent_classes = {
        "SAC": SACAgent,
        "SACGC": SACAgentGC,
        "TD3": TD3Agent,
        "TD3GC": TD3AgentGC,
        "CrossQ": CrossQAgent,
    }

    agent_name = cfg.algorithm.agent_name
    if agent_name not in agent_classes:
        raise ValueError(f"Unknown agent name: {agent_name}. Available: {list(agent_classes.keys())}")

    SelectedAgentClass = agent_classes[agent_name]
    print(f"Using agent: {agent_name}")

    # Create the selected agent
    agent = SelectedAgentClass.create(
        seed=cfg.seed,
        observations=sample_obs,
        actions=sample_act,
        agent_config=agent_config
    )
    # --- End Agent Initialization ---

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        capacity=cfg.training.replay_buffer_capacity
    )

    # Training loop
    observation, _ = env.reset(seed=cfg.seed)
    for step_num in tqdm(range(1, cfg.training.max_steps + 1), desc=f"Training {cfg.environment.name}"):
        if step_num < cfg.training.start_steps:
            # Sample random actions before training starts
            action = env.action_space.sample()
        else:
            # Sample actions from the policy using agent.sample
            # agent.sample returns (new_agent_with_updated_rng, action)
            agent, action = agent.sample(observation)

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
                            'train/episode_length': info['episode']['l'][0]},
                            step=step_num, commit=False)

        # Perform agent update if enough steps have passed
        if step_num >= cfg.training.start_steps:

            for i in range(cfg.training.update_frequency):

                batch = replay_buffer.sample(cfg.training.batch_size)
                agent, update_info = agent.update(batch)

            if cfg.logging.use_wandb and update_info:
                wandb.log({f'train/{k}': v for k, v in update_info.items()},step=step_num, commit=False)

        # Evaluate agent periodically
        if step_num % cfg.training.eval_freq == 0:
            eval_returns = []
            for _ in range(cfg.training.eval_episodes):
                
                 agent, avg_reward = evaluate(agent, eval_env, 1)
                 eval_returns.append(avg_reward)


            avg_eval_reward = np.mean(eval_returns)
            # Log evaluation results to wandb
            log_data = {"eval/avg_reward": avg_eval_reward, "step": step_num}
            if cfg.logging.use_wandb:
                wandb.log(log_data,step=step_num, commit=True) # Commit updates here
          

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