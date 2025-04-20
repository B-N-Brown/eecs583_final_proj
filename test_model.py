import os
import gym
import torch
import compiler_gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from compiler_gym.wrappers import RewardWrapper
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag

from runtime_reward import RuntimeInstCountRewardWrapper

import stable_baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_opt_actions, loop_action_space


MAX_INFERENCE_ITERS = 20 


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

def test_model_loop(env, checkpoint_name="basic_model.pth", fig_sub_dir="basic_figs"):
    # benchmarks = env.datasets["cbench-v1/sha"].benchmarks # TODO: choose a benchmark that actually has loops
    total_rewards = []
    initial_execution_times = []
    final_execution_times = []
    prediction_times = []
    reward_ratios = []
    action_selection_freq = {} 

    # TODO: how to tune model architecture for PPO?
    model = PPO.load(f"model_checkpoints/{checkpoint_name}", 
                     print_system_info=True) # TODO: consider using something other than PPO? 

    n_episodes = 25 # TODO: Hyperparameter tuning
    done = False

    for episode in tqdm(range(n_episodes)):
        env.reset()
        total_reward = 0
        observation = env.reset()

        # Getting baseline (initial) run
        observation, reward, done, info = env.step([])
        initial_execution_times.append(env.reward.spaces['runtime'].previous_runtime)

        with tqdm(desc="Processing") as pbar:
            tqdm_counter = 0

            while not done:
                if tqdm_counter > MAX_INFERENCE_ITERS:
                    print("HIT MAX ITER WALL")
                    break

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
                    
                    if action_name not in action_selection_freq:
                        action_selection_freq[env.action_space.to_string(action)] = 0

                    else:
                        action_selection_freq[env.action_space.to_string(action)] += 1

                total_reward += reward
                    
                tqdm_counter += 1
                pbar.update(1)

        
        # Final data collection 
        # print("Time to run inference on episode:", env.episode_walltime)
        total_rewards.append(total_reward)
        prediction_times.append(env.episode_walltime)
        # reward_ratios.append(env.reward_ratio)
        final_execution_times.append(env.reward.spaces['runtime'].previous_runtime)


    # Plotting all the results
    data_vis(
        total_rewards,
        prediction_times, 
        reward_ratios, 
        final_execution_times, 
        initial_execution_times,
        action_selection_freq,
        fig_sub_dir
    )


def data_vis(
        total_rewards,
        prediction_times, 
        reward_ratios, 
        final_execution_times, 
        initial_execution_times,
        action_selection_freq,
        fig_sub_dir="basic_figs"
) -> None:
    assert(len(final_execution_times) > 0)

    if not os.path.isdir(f"figs/{fig_sub_dir}"):
        os.mkdir(f"figs/{fig_sub_dir}")

    fig, ax = plt.subplots()

    ax.plot(total_rewards, "*")
    ax.set_title("Total Episode Rewards (Exec Time)")
    plt.savefig(f"figs/{fig_sub_dir}/total_rewards.png")
    fig.clf()

    fig, ax = plt.subplots()

    ax.plot(prediction_times)
    ax.set_title("Time to optimize (inference time)")
    plt.savefig(f"figs/{fig_sub_dir}/infer_time.png")
    fig.clf()

    fig, ax = plt.subplots()
    ax.plot(reward_ratios)
    ax.set_title("Reward Ratio")
    plt.savefig(f"figs/{fig_sub_dir}/reward_ratio.png")
    fig.clf()

    fig, ax = plt.subplots()
    ax.plot(final_execution_times, "*", c="b", label="Final")
    ax.plot(initial_execution_times, "+", c="r", label="Initial")
    ax.set_title("Execution time before/after optimization")
    ax.legend()
    plt.savefig(f"figs/{fig_sub_dir}/exec_time.png")
    fig.clf()

    fig, ax = plt.subplots()
    ax.plot(np.array(initial_execution_times) - np.array(final_execution_times), "*", c="b", label="Final")
    # ax.scatter(initial_execution_times, "+", c="r", label="Initial")
    ax.set_title("Execution time before/after optimization (difference)")
    ax.legend()
    plt.savefig(f"figs/{fig_sub_dir}/exec_time_diff.png")
    fig.clf()


    fig, ax = plt.subplots()
    bins = action_selection_freq.keys()
    freqs = action_selection_freq.values()
    plt.title("Instruction selection frequencies")
    plt.bar(bins, freqs)
    plt.savefig(f"figs/{fig_sub_dir}/instr_hist.png")


def main():
    env = compiler_gym.make(
        "llvm-v0",
        benchmark="cbench-v1/qsort", # TODO: find a good benchmark
        observation_space="Autophase", # TODO: consider using a different observation space that maybe isn't as broad?
        reward_space="IrInstructionCountOz"
    )

    env.action_space = loop_action_space
    env.reset()
    env = RuntimeInstCountRewardWrapper(env)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Testing code
    # test_model_loop(env, checkpoint_name="a2c_mlp_50epochs.pth")
    # test_model_loop(env, checkpoint_name="mlp_cbench_26rollout.pth")
    # test_model_loop(env, checkpoint_name="mlp_cbench_26rollout_30steps_20batch_rewardfix.pth", 
    #                 fig_sub_dir="mlp_cbench_26rollout_30steps_20batch_rewardfix")
    
    test_model_loop(env, checkpoint_name="mlp_cbench_26rollout_30steps_20batch_lr_0002_rewardfix.pth", 
                    fig_sub_dir="mlp_cbench_26rollout_30steps_20batch_lr_0001_rewardfix.pth")




if __name__ == "__main__":
    main()
