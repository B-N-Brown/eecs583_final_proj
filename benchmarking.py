import compiler_gym
from runtime_reward import RuntimeInstCountRewardWrapper
import matplotlib.pyplot as plt
import numpy as np
import gym
from tqdm import tqdm

from compiler_gym.spaces.commandline import Commandline, CommandlineFlag

def evaluate_with_flag(benchmark_uri: str, flag: str):
    
    env = compiler_gym.make(
        "llvm-v0",
        benchmark=benchmark_uri,
        observation_space="Autophase",
        reward_space="IrInstructionCountOz"
    )
    env.reset()
    env = RuntimeInstCountRewardWrapper(env)

    # env.observation_space = ["Runtime", "IrInstructionCount"]
    # print(env.observation.spaces.keys())
    
    # observation, reward, done, info = env.apply(
    #     [flag], observation_space=["Runtime", "IrInstructionCount"]
    # )

    flag = CommandlineFlag(flag, flag, flag)
    env.action_space = Commandline([flag])
    action = env.action_space.sample()
    # print("ACTION SHOULD BE THE SAM:", action)
    env.step(action)

    runtime = env.reward.spaces['runtime'].previous_runtime
    inst_count = env.reward.spaces['runtime'].previous_inst_count

    # runtime = np.median(observation["Runtime"])
    # inst_count = observation["IrInstructionCount"]

    env.close()
    return runtime, inst_count


optimization_flags = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"]
runnable_benchmarks = ['benchmark://cbench-v1/bitcount', 'benchmark://cbench-v1/blowfish', 'benchmark://cbench-v1/bzip2', 'benchmark://cbench-v1/crc32', 'benchmark://cbench-v1/dijkstra', 'benchmark://cbench-v1/gsm', 'benchmark://cbench-v1/jpeg-c', 'benchmark://cbench-v1/jpeg-d', 'benchmark://cbench-v1/patricia', 'benchmark://cbench-v1/qsort', 'benchmark://cbench-v1/sha', 'benchmark://cbench-v1/stringsearch', 'benchmark://cbench-v1/stringsearch2', 'benchmark://cbench-v1/susan', 'benchmark://cbench-v1/tiff2bw', 'benchmark://cbench-v1/tiff2rgba', 'benchmark://cbench-v1/tiffdither', 'benchmark://cbench-v1/tiffmedian']

results = {}

for benchmark in tqdm(runnable_benchmarks):
    results[benchmark] = {}
    for flag in optimization_flags:
        try:
            runtime, inst_count = evaluate_with_flag(benchmark, flag)
        except Exception as e:
            print(f"Error on {benchmark} with {flag}: {e}")
            runtime, inst_count = float("inf"), -1
        results[benchmark][flag] = {"runtime": runtime, "inst_count": inst_count}

for benchmark in results:
    runtimes = [results[benchmark][f]["runtime"] for f in optimization_flags]
    
    plt.figure()
    plt.bar(optimization_flags, runtimes)
    plt.title(f"Runtime for {benchmark}")
    plt.ylabel("Median Runtime")
    plt.xlabel("Optimization Flag")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{benchmark}.png')

