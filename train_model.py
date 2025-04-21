"""
Simple policy training script and makeshift environment sampling loop

TODO List:
    * Investigate different datasets --> seems like cbench isn't suited for loop unrolling?
    * Try different feature vectors. Autophase might be too big
    * Evaluate different learning frameworks:
        - A2C
        - DQN
        - Others?
    * Finetune ppo params (how tf does this work?)
"""
import torch
import os
import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks, IterateOverBenchmarks, ConstrainedCommandline
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_action_space, loop_opt_actions

from runtime_reward import RuntimeInstCountRewardWrapper
from dataset_wrapper import CBenchWrapper

# STABLE BASELINES TRAINING REGIMES
def ppo_training_sb(env, device, checkpoint_name="basic_model.pth"):
    n_steps = 30 
    n_envs = 1
    total_timesteps = n_steps * n_envs # PPO training min
    total_timesteps *= 16 # Total number of rollouts

    # Create vectorized env (recommended for SB3)
    vec_env = make_vec_env(lambda: env, n_envs=n_envs)
   
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=2, 
        batch_size=256,
        n_steps=2048,
        n_epochs=50,
        ent_coef=1e-1,
        learning_rate=1e-4,
        normalize_advantage=True,
        seed=42,
        device=device,
        #tensorboard_log="/home/bnb/Documents/uofm/eecs583/final_proj/tensor_boards"
    )
    
    checkpoint_dir = "johns_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1 ,   # save every 50k env‑steps total
        save_path=checkpoint_dir,
        name_prefix="ppo_compilergym"
    )

    # (Optional) also evaluate periodically and save the best model
    eval_callback = EvalCallback(
        env, 
        best_model_save_path="johns_checkpoints",
        log_path="./logs/results/",
        eval_freq=1,
        deterministic=True,
        render=False
    )

    # 4) Combine callbacks and start learning
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=10, callback=callbacks, progress_bar=True)
    model.save(path=f"model_checkpoints/{checkpoint_name}")

def a2c_training_sb(env, checkpoint_name="basic_model.pth"):
    total_timesteps = 2048

    # Wrap in a vectorized env (required by SB3)
    vec_env = make_vec_env(lambda: env, n_envs=8)

    # Optional: custom network
    # policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # Train
    model = A2C("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(path=f"model_checkpoints/{checkpoint_name}")


def main():
    # Checking out the dataset
    # compiler_gym.envs.llvm.datasets.CBenchDataset("cbench-v1/sha")

    # Uncomment to install all datasets (takes a couple mins)
    # print("PRINTING:", compiler_gym.envs.llvm.datasets.get_llvm_datasets())
    # for dset in compiler_gym.envs.llvm.datasets.get_llvm_datasets():
    #     print(dset.install())

    # angha_dset = compiler_gym.envs.llvm.datasets.AnghaBenchDataset("anghabench-v1")
    # cbench_dset = compiler_gym.envs.llvm.datasets.CBenchDataset("cbench")

    # dataset = env.datasets["cbench-v1"]

    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )
    runnable_benchmarks = ['benchmark://cbench-v1/bitcount', 'benchmark://cbench-v1/blowfish', 'benchmark://cbench-v1/bzip2', 'benchmark://cbench-v1/crc32', 'benchmark://cbench-v1/dijkstra', 'benchmark://cbench-v1/gsm', 'benchmark://cbench-v1/jpeg-c', 'benchmark://cbench-v1/jpeg-d', 'benchmark://cbench-v1/patricia', 'benchmark://cbench-v1/qsort', 'benchmark://cbench-v1/sha', 'benchmark://cbench-v1/stringsearch', 'benchmark://cbench-v1/stringsearch2', 'benchmark://cbench-v1/susan', 'benchmark://cbench-v1/tiff2bw', 'benchmark://cbench-v1/tiff2rgba', 'benchmark://cbench-v1/tiffdither', 'benchmark://cbench-v1/tiffmedian']
    # runnable_benchmarks = []
    if len(runnable_benchmarks) == 0:
        for benchmark in dataset.benchmark_uris():
            count += 1
            try:
                env.reset(benchmark=benchmark)
                if len(env.observation['Runtime']) > 0:
                    runnable_benchmarks.append(benchmark)
            except:
                continue
    env.reset()
    
    # Restrict action space to loop actions
    env.action_space = loop_action_space

    env = CycleOverBenchmarks(env, runnable_benchmarks)
    env = RuntimeInstCountRewardWrapper(env)
    env.reset()
    seed = 42

    env.action_space.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    # Train model on MLP policy network
    # basic_train(env)

    ppo_training_sb(env, device, checkpoint_name="mlp_cbench_32rollout_30steps_20batch_lr_0001_rewardfix.pth")
    # a2c_training_sb(env, checkpoint_name="a2c_mlp_50epochs.pth")
    env.close()


if __name__ == "__main__":
    main()
