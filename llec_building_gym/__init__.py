# llec_building_gym/__init__.py

# Gymnasium registration for custom environments
import gymnasium as gym
from gymnasium.envs.registration import register

# Import core components from the llec_building_gym package
from llec_building_gym.envs import (
    BaseBuildingGym,
    Building,
    FuzzyController,
    MPCController,
    PIController,
    PIDController,
)
# Register environment with temperature reward mode
register(
    id="LLEC-HeatPumpHouse-1R1C-Temperature-v0",
    entry_point="llec_building_gym.envs:BaseBuildingGym",
    max_episode_steps=288,
    kwargs={
        "reward_mode": "temperature", # temperature-based reward
        "render_mode": None,  
    },
)

# Register environment with combined reward mode
register(
    id="LLEC-HeatPumpHouse-1R1C-Combined-v0",
    entry_point="llec_building_gym.envs:BaseBuildingGym",
    max_episode_steps=288,
    kwargs={
        "reward_mode": "combined", # temperature and economic reward
        "temperature_weight": 1.0,
        "economic_weight": 1.0,
        "render_mode": None,  
    },
)

# Exported components of the llec_building_gym package
__all__ = [
    "BaseBuildingGym",
    "Building",
    "FuzzyController",
    "MPCController",
    "PIController",
    "PIDController",
]

