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
import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks, IterateOverBenchmarks, ConstrainedCommandline
# from stable_baselines3.common.logger import configure

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from loop_actions import loop_action_space, loop_opt_actions

from runtime_reward import RuntimeInstCountRewardWrapper
from dataset_wrapper import CBenchWrapper

# STABLE BASELINES TRAINING REGIMES
def ppo_training_sb(env, device, checkpoint_name="basic_model.pth"):
    n_steps = 15 
    n_envs = 2 
    total_timesteps = n_steps * n_envs # PPO training min
    total_timesteps *= 26 # Total number of rollouts

    # Create vectorized env (recommended for SB3)
    vec_env = make_vec_env(lambda: env, n_envs=n_envs)
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=2, 
        batch_size=10,
        n_steps=n_steps,
        n_epochs=50,
        learning_rate=0.0002,
        normalize_advantage=True,
        seed=42,
        device=device,
        tensorboard_log="/home/bnb/Documents/uofm/eecs583/final_proj/tensor_boards"
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
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

    ppo_training_sb(env, device, checkpoint_name="test_alpha.pth")
    env.close()


if __name__ == "__main__":
    main()
