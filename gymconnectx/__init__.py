# gymconnectx/__init__.py

import gym
import pygame
from gym.envs.registration import register

# Register the custom environment
register(
    id='gymconnectx/ConnectGameEnv',
    entry_point='gymconnectx.envs:ConnectGameEnv',
    kwargs={
        'connect': 4,
        'width': 7,
        'height': 7,
        'reward_winner': 1,
        'reward_loser': -1,
        'living_reward': 0,
        'max_steps': 100,
        'delay': 100,
        'square_size': 100,
        'avatar_player_1': None,
        'avatar_player_2': None
    }
)

from . import envs

