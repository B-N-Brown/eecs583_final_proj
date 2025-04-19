import gym
import torch
import compiler_gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from compiler_gym.wrappers import RewardWrapper
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag

from runtime_reward import RuntimeImprovementWrapper

import stable_baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_opt_actions, loop_action_space


def test_model(env, checkpoint_name="basic_model.pth"):
    # model = model.load(path=f"model_checkpoints/{checkpoint_name}")

    model = PPO.load(f"model_checkpoints/{checkpoint_name}", 
                     print_system_info=True)
    
    averaged_reward = evaluate_policy(
                            env=env, 
                            model=model,
                            n_eval_episodes=1,
                            reward_threshold=0.0000001,
                            return_episode_rewards=True,              
                     ) 

    print("Averaged reward:", averaged_reward)

def test_model_loop(env, checkpoint_name="basic_model.pth"):
    # benchmarks = env.datasets["cbench-v1/sha"].benchmarks
    total_rewards = []
    initial_execution_times = []
    final_execution_times = []
    prediction_times = []
    action_selction_freq = {} 

    empty_flag = CommandlineFlag(" ", " ", "empty_flag")

    model = PPO.load(f"model_checkpoints/{checkpoint_name}", 
                     print_system_info=True)


    n_episodes = 1 
    # env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)
    done = False

    for episode in (range(n_episodes)):
        env.reset()
        total_reward = 0
        observation = env.reset()
        print("obs:", observation)

        # Getting baselines run
        # env.action_space = empty_flag
        # fake_action = env.sam
        # env.step

        # with tqdm(desc="Processing") as pbar:
        #     tqdm_counter = 0

        while not done:
            print("iterating through done loop")
            action, _ = model.predict(observation, deterministic=False)
            action = int(action.tolist())
            
            #print("SAMPLD ACTION:", env.action_space.to_string(action))

            # action = env.action_space.sample()
            # print("ACTION:", action)

            observation, reward, done, info = env.step(action)

            # print("INFO:", info)
            # print("REWARD: ", reward)

            if not info["action_had_no_effect"]:
                action_name = env.action_space.to_string(action)
                
                if action_name not in action_selction_freq:
                    action_selction_freq[env.action_space.to_string(action)] = 0

                else:
                    action_selction_freq[env.action_space.to_string(action)] += 1

            total_reward += reward
                
                # tqdm_counter += 1
                # pbar.update(1)
            print()
            break

                # Done threshold
            
            # exit()

        print("Time to run inference on episode:", env.episode_walltime)
        total_rewards.append(total_reward)
        prediction_times.append(env.episode_walltime)

    # plt.plot(total_reward)
    # plt.savefig("test_plot.png")

def main():
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/sha", 
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )

    env.action_space = loop_action_space

    env = RuntimeImprovementWrapper(env)
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Testing code
    test_model_loop(env)




if __name__ == "__main__":
    main()
