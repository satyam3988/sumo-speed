# import os
# import sys
# import gymnasium
# from stable_baselines3.dqn.dqn import DQN
# from stable_baselines3.common.callbacks import BaseCallback
# from sumo_rl import SumoEnvironment
# import traci

# class RewardLoggerCallback(BaseCallback):
#     def _on_step(self) -> bool:
#         print("Callback executed!")  # Debug print
#         if "infos" in self.locals.keys():
#             print(self.locals.get("infos"))  # Print entire 'infos' structure
#             rewards = [info.get('reward') for info in self.locals.get("infos") if 'reward' in info]
#             for reward in rewards:
#                 print(f"Reward: {reward}")
#         return True
    

# def setup_sumo_home():
#     if "SUMO_HOME" in os.environ:
#         tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#         sys.path.append(tools)
#     else:
#         sys.exit("Please declare the environment variable 'SUMO_HOME'")

# # if "SUMO_HOME" in os.environ:
# #     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
# #     sys.path.append(tools)
# # else:
# #     sys.exit("Please declare the environment variable 'SUMO_HOME'")


# # from sumo_rl import SumoEnvironment
# if __name__ == "__main__":
#     print("Hlleo")
#     setup_sumo_home()

#     env = SumoEnvironment(
#         net_file="nets/big-intersection/big-intersection.net.xml",
#         single_agent=True,
#         route_file="nets/big-intersection/routes.rou.xml",
#         out_csv_name="outputs/big-intersection/dqn",
#         use_gui=True,
#         num_seconds=5400,
#         # yellow_time=4,
#         # min_green=10,
#         # max_green=60,
#     )

#     model = DQN(
#         env=env,
#         policy="MlpPolicy",
#         learning_rate=1e-3,
#         learning_starts=0,
#         buffer_size=50000,
#         train_freq=1,
#         target_update_interval=500,
#         exploration_fraction=0.05,
#         exploration_final_eps=0.01,
#         verbose=1,
#     )
#     num_episodes = 10
#     timesteps_per_episode = env.num_seconds if hasattr(env, "num_seconds") else 1000
#     total_timesteps = num_episodes * timesteps_per_episode

#     model.learn(total_timesteps=total_timesteps)
#     model.learn(total_timesteps=10000)
#     model.save("saved_models/big-intersection/dqn/model1")

# Below this is the code for testing phase using the model that has been saved

import os
import sys
import gymnasium
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment
import numpy as np
import traci
import csv

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
    setup_sumo_home()

    env = SumoEnvironment(
        net_file="nets/big-intersection/big-intersection.net.xml",
        single_agent=True,
        route_file="nets/big-intersection/routes.rou.xml",
        out_csv_name="outputs/big-intersection/dqn",
        use_gui=True,
        num_seconds=5400,
        # yellow_time=4,
        # min_green=10,
        # max_green=60,
        # num_seconds=40000,
    )

    # Load the saved model
    model = DQN.load("saved_models/big-intersection/dqn/model1")

    # Use the loaded model in the environment
    obs, info = env.reset()
    print("Observation structure:", obs)
    total_reward = 0
    num_steps = 0
    system_mean_waiting_time = 0

    # Create a CSV file and write headers
    with open('outputs/big-intersection/dqn/test/output.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Steps', 'system_mean_waiting_time'])

        for step in range(1500):
            action, _states = model.predict(obs, deterministic=True)
            results = env.step(action)
            obs = results[0]
            reward = results[1]
            done = results[2]
            info = results[4]
            print("Info: ", info)
            
            # Update metrics
            total_reward += reward
            num_steps += 1
            if isinstance(info, dict):  # Check if info is a dictionary
                system_mean_waiting_time = info.get('system_mean_waiting_time', 0)

            # Write data to CSV
            csvwriter.writerow([step, system_mean_waiting_time])

            if done:
                obs = env.reset()

    print("Total reward over 100 steps:", total_reward)
    print("Average reward per step:", total_reward / num_steps)
    print("Average system mean waiting time per step:", system_mean_waiting_time / num_steps)

    # If you want to continue training the loaded model, uncomment the lines below:
    # num_episodes = 12
    # timesteps_per_episode = env.num_seconds if hasattr(env, "num_seconds") else 1000
    # total_timesteps = num_episodes * timesteps_per_episode
    # model.learn(total_timesteps=total_timesteps)
