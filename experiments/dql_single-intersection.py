import os
import sys
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment
import traci

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

if __name__ == "__main__":
    print("Hello")
    setup_sumo_home()

    env = SumoEnvironment(
        net_file="nets/single-intersection/single-intersection.net.xml",
        route_file="nets/single-intersection/single-intersection.rou.xml",
        out_csv_name="outputs/single-intersection/double_ql/d",
        single_agent=True,
        use_gui=True,
        num_seconds=40000,
    )

    print("Environment setup done.")

    # Initialize the DQN model
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=0  # Suppress verbose output
    )

    num_episodes = 10
    timesteps_per_episode = env.num_seconds if hasattr(env, "num_seconds") else 1000
    total_timesteps = num_episodes * timesteps_per_episode

    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    # model.save("saved_models/single-intersection/dqn/max_60")
