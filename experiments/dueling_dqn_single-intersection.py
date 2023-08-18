
import traci
from sumo_rl import SumoEnvironment
import os
import sys
import gymnasium as gym
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.dqn import DQN
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQN, self).__init__()
        
        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages to get Q-values
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class CustomDuelingNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super(CustomDuelingNetwork, self).__init__(observation_space, features_dim)
        num_inputs = observation_space.shape[0]
        num_outputs = features_dim
        self.dueling_dqn = DuelingDQN(num_inputs, num_outputs)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.dueling_dqn(observations)

class DuelingDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomDuelingNetwork)


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="nets/single-intersection/single-intersection.net.xml",
        route_file="nets/single-intersection/single-intersection.rou.xml",
        out_csv_name="outputs/single-intersection/new_dueling_dqn/d",
        single_agent=True,
        use_gui=True,
        num_seconds=20000,
    )
    model = DQN(
        env=env,
        policy=DuelingDQNPolicy,
        learning_rate=0.0005,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=500,
        exploration_initial_eps=0.2,
        exploration_final_eps=0.01,
        verbose=1,
    )
    num_episodes = 50
    if hasattr(env, "num_seconds"):
        timesteps_per_episode = env.num_seconds
    else:
        timesteps_per_episode = 1000
    total_timesteps = num_episodes * timesteps_per_episode
    model.learn(total_timesteps=total_timesteps)