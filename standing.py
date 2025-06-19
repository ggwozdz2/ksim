# mypy: disable-error-code="override"
"""Defines simple task for training a standing policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar
import xax
import ksim
from .rewards.standing_reward import HumanoidStandingTaskConfig, HumanoidStandingTask

if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.standing
    # To visualize the environment, use the following command:
    #  python -m examples.standing run_mode=view batch_size=1 num_envs=1
    #   python -m examples.standing run_mode=view
    HumanoidStandingTask.launch(
        HumanoidStandingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=32,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
        ),
    )
