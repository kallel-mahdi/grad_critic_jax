import itertools
import os
import subprocess
import sys

import hydra
import submitit
from omegaconf import DictConfig


def run_experiment(algorithm_exec_file: str, environment: str, seed: int, algorithm: str, update_frequency: int, wandb_project: str) -> None:
    """
    Runs a single experiment via a subprocess run, passing the full inherited environment.

    Args:
        algorithm_exec_file: Algorithm/system exec file (e.g., 'main.py').
        environment: Environment config (e.g. 'mujoco/Ant-v4')
        seed: Random seed for reproducibility
        algorithm: Name of the algorithm (e.g., 'SAC', 'TD3GC')
    """
    print(f"--- Starting experiment: {algorithm} on {environment} with seed {seed} update_frequency={update_frequency} wandb_project={wandb_project} ({algorithm_exec_file}) ---")
    print(f"Launcher CWD: {os.getcwd()}")

    # Assuming python is in .venv/bin relative to the job's working directory
    cmd = f".venv/bin/python {algorithm_exec_file} environment={environment} seed={seed} algorithm={algorithm} \
                            training.update_frequency={update_frequency} logging.wandb_project={wandb_project}"
    print(f"Executing command: {cmd}")

    # Pass a copy of the current environment to the subprocess.
    # This ensures variables set by SLURM/submitit setup are inherited.
    process_env = os.environ.copy()

    # Print HTTPS_PROXY environment variable
    print(f"HTTPS_PROXY environment variable: {process_env.get('HTTPS_PROXY', 'Not set')}")

    process = subprocess.run(
        cmd,
        shell=True,        # Runs command through the shell
        check=True,        # Raises CalledProcessError on non-zero exit code
        text=True,         # Decodes stdout/stderr as text (though not captured here)
        timeout=None,      # No timeout for the subprocess
        env=process_env    # Pass the full captured environment
    )


def filter_none_values(d: dict) -> dict:
    """
    Returns a new dictionary containing only the items from the input dictionary
    where the value is not None.

    Args:
        d: The input dictionary.
    Returns:
        A dictionary with keys whose values are not None.
    """
    return {key: value for key, value in d.items() if value is not None}


@hydra.main(version_base="1.2", config_path="./configs/launcher", config_name="slurm")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for launching multiple Stoix experiments on SLURM-based cluster.

    Args:
        cfg: The Hydra-populated configuration object.
    """
    # Create the submitit executor for SLURM.
    executor = submitit.AutoExecutor(folder=cfg.slurm.folder)

    # Build SLURM parameter dictionary for submitit executor.
    slurm_params_direct = {
        "nodes": cfg.slurm.nodes,
        "gpus_per_node": cfg.slurm.gpus_per_node,
        "cpus_per_task": cfg.slurm.cpus_per_task,
        "gres": cfg.slurm.gres,
        "time": cfg.slurm.time,
        "mem_gb": cfg.slurm.get("mem_gb"), # Use .get for optional parameter
        "slurm_partition": cfg.slurm.partition,
        "slurm_account": cfg.slurm.account,
        "slurm_qos": cfg.slurm.qos,

    }
    # Filter out any None values the user might leave in the config
    slurm_params_direct = filter_none_values(slurm_params_direct)

    # Build dictionary for slurm_additional_parameters
    slurm_params_additional = {
        "export": cfg.slurm.export,  # Direct mapping without .get()
        # Add any other SBATCH directives here that don't have direct submitit args
    }
    slurm_params_additional = filter_none_values(slurm_params_additional)


    # Update the executor with SLURM parameters
    executor.update_parameters(
        slurm_job_name=cfg.experiment_group,
        slurm_setup=list(cfg.slurm.setup_commands), # Use 'setup' to run on SLURM node
        slurm_additional_parameters=slurm_params_additional,
        **slurm_params_direct # Pass direct parameters like time, mem_gb, etc.
    )

    # ---------------------------------------------------------------------------
    jobs = []
    # Prepare the Cartesian product of algorithm_execs, environments, seeds, algorithms.
    for algorithm_exec, environment, seed, alg_name, update_frequency, wandb_project in itertools.product(
        cfg.experiment.algorithm_exec_files,
        cfg.experiment.environments,
        cfg.experiment.seeds,
        cfg.experiment.algorithm,
        cfg.experiment.update_frequency,
        cfg.experiment.wandb_project
    ):
        print(f"Submitting independent job for {alg_name} on {environment} seed={seed}")
        job = executor.submit(run_experiment, algorithm_exec, environment, seed, alg_name, update_frequency, wandb_project)
        print(f" -> SLURM Job ID: {job.job_id}")
        jobs.append(job)
    # ---------------------------------------------------------------------------

    print(f"Launched {len(jobs)} tasks.")
    print("All SLURM job IDs:")
    for job in jobs:
        print(f" - {job.job_id}")


if __name__ == "__main__":
    main()