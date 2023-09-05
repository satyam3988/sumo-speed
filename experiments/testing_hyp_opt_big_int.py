import os
import sys
import numpy as np
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment
import traci
from hyperopt import fmin, tpe, hp

class RewardLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        print("Callback executed!")  # Debug print
        if "infos" in self.locals.keys():
            print(self.locals.get("infos"))  # Print entire 'infos' structure
            rewards = [info.get('reward') for info in self.locals.get("infos") if 'reward' in info]
            for reward in rewards:
                print(f"Reward: {reward}")
        return True
    

def setup_sumo_home():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")


# Define hyperopt space
space = {
    'learning_rate': hp.loguniform('learning_rate', -6.9, -4.6),
    'buffer_size': hp.choice('buffer_size', [10000, 50000, 100000, 150000]),
    'train_freq': hp.choice('train_freq', [1, 2, 4, 8]),
    'target_update_interval': hp.choice('target_update_interval', [100, 500, 1000]),
    'exploration_fraction': hp.uniform('exploration_fraction', 0.01, 0.1),
    'exploration_final_eps': hp.uniform('exploration_final_eps', 0.005, 0.05)
}

# Objective function for hyperopt with tuple observation handling and step results extraction
def objective(params):
    model = DQN(
        env=env,
        policy="MlpPolicy",
        **params,
        verbose=0
    )
    mean_reward = 0
    for _ in range(num_episodes):
        results = env.reset()
        obs = results[0]
        # Ensure observation shape is (37,)
        if obs.ndim == 0:
            obs = obs.reshape(1, -1)
        obs = obs.squeeze()
        episode_reward = 0
        for _ in range(timesteps_per_episode):
            action, _states = model.predict(obs)
            
            results = env.step(action)
            obs = results[0]
            reward = results[1]
            done = results[2]
            info = results[3]

            # Ensure observation shape is (37,)
            if obs.ndim == 0:
                obs = obs.reshape(1, -1)
            obs = obs.squeeze()
            episode_reward += reward
            if done:
                break
        mean_reward += episode_reward
    mean_reward /= num_episodes
    return -mean_reward


if __name__ == "__main__":
    print("Hello")
    setup_sumo_home()

    env = SumoEnvironment(
        net_file="nets/2way-single-intersection/single-intersection.net.xml",
        single_agent=True,
        route_file="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn/hyp",
        use_gui=True,
        num_seconds=5400,
    )

    num_episodes = 1
    timesteps_per_episode = env.num_seconds if hasattr(env, "num_seconds") else 1000

    # Using hyperopt to find the best hyperparameters
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50
    )
    print("Best hyperparameters:", best)

    # Train the model using the best hyperparameters
    model = DQN(
        env=env,
        policy="MlpPolicy",
        **best,
        verbose=1
    )
    total_timesteps = num_episodes * timesteps_per_episode
    model.learn(total_timesteps=total_timesteps)
    # model.save("saved_models/big-intersection/dqn/model1")



