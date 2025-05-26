"""Defines a task for training a policy using AMP, building on PPO."""

__all__ = [
    "AMPConfig",
    "AMPTask",
    "AMPReward",
]

import bdb
import itertools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Iterable, TypeVar

import attrs
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import tqdm
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import DictConfig, OmegaConf

from ksim.debugging import JitLevel
from ksim.rewards import Reward
from ksim.task.ppo import PPOConfig, PPOTask, PPOVariables
from ksim.task.rl import (
    LoggedTrajectory,
    RewardState,
    RLLoopCarry,
    RLLoopConstants,
    RolloutConstants,
    RolloutEnvState,
    RolloutSharedState,
    get_viewer,
)
from ksim.types import PhysicsModel, Trajectory

DISCRIMINATOR_OUTPUT_KEY = "_discriminator_output"
REAL_MOTIONS_KEY = "_real_motions"

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class EurekaConfig(PPOConfig):
    """Configuration for Adversarial Motion Prior training."""

    # Toggle this to visualize the motion on the robot.
    run_motion_viewer: bool = xax.field(
        value=False,
        help="If true, the motion will be visualized on the robot.",
    )
    run_motion_viewer_loop: bool = xax.field(
        value=True,
        help="If true, the motion will be looped.",
    )


Config = TypeVar("Config", bound=EurekaConfig)


# Can we run it on the go or load up each time? - 
class EurekaTask(PPOTask[Config], Generic[Config], ABC):
    """Adversarial Motion Prior task.

    This task extends PPO to include adversarial training with a discriminator
    that tries to distinguish between real motion data and policy-generated motion.
    """

    def reward_reflection(self, trajectory: Trajectory) -> Array:
        """Reward reflection.

        This function is used to reflect the reward of the trajectory.
        """
        return trajectory.rewards
    
    def evaluate_policies(self, policies: list[PyTree]) -> Array:
        pass

    def reward_reflection(self, trajectory: Trajectory) -> Array:
        pass