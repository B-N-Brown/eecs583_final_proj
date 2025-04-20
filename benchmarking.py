import compiler_gym
from runtime_reward import RuntimeInstCountRewardWrapper
import matplotlib.pyplot as plt
import numpy as np
import gym
from tqdm import tqdm
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag

# def evaluate_with_flag(benchmark_uri: str, flag: str):
#     env = compiler_gym.make(
#         "llvm-v0",
#         observation_space="Autophase",
#         reward_space="IrInstructionCountOz"
#     )

#     # print("Before:", env.observation.spaces.keys())
#     # env.observation.add_space("Runtime")
#     # env.observation.add_space("IrInstructionCount")
#     # print("After:", env.observation.spaces.keys())
    
#     env = RuntimeInstCountRewardWrapper(env)

#     try:
#         env.observation_space = ["Runtime", "IrInstructionCount"]
#         env.reset(benchmark=benchmark_uri)

#         print(f'Observation Spaces Available: {env.observation.spaces.keys()}')

#         if not env.observation["IsRunnable"]:
#             raise RuntimeError(f"Benchmark {benchmark_uri} is not runnable")

#         observation, reward, done, info = env.apply(
#             [flag], observation_spaces=["Runtime", "IrInstructionCount"]
#         )

#         runtime = np.median(observation["Runtime"])
#         inst_count = observation["IrInstructionCount"]
#     except Exception as e:
#         print(f"Error on {benchmark_uri} with {flag}: {e}")

#     runtime, inst_count = float("inf"), -1


#     env.close()
#     return runtime, inst_count
def evaluate_with_flag(benchmark_uri: str, flag: str):
    env = compiler_gym.make(
        "llvm-v0", 
        observation_space='Autophase', 
        reward_space="IrInstructionCountOz"
        )

    try:
        env.reset(benchmark=benchmark_uri)

        observation, _, _, _ = env.apply(
            [flag], observation_spaces=["Runtime", "IrInstructionCount"]
        )

        runtime = np.median(observation["Runtime"])
        inst_count = observation["IrInstructionCount"]

    except Exception as e:
        print(f"Error on {benchmark_uri} with {flag}: {e}")
        runtime, inst_count = float("inf"), -1

    env.close()
    return runtime, inst_count



optimization_flags = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"]
# optimization_flags = ["-O0"]
runnable_benchmarks = ['benchmark://cbench-v1/bitcount', 'benchmark://cbench-v1/blowfish', 'benchmark://cbench-v1/bzip2', 'benchmark://cbench-v1/crc32', 'benchmark://cbench-v1/dijkstra', 'benchmark://cbench-v1/gsm', 'benchmark://cbench-v1/jpeg-c', 'benchmark://cbench-v1/jpeg-d', 'benchmark://cbench-v1/patricia', 'benchmark://cbench-v1/qsort', 'benchmark://cbench-v1/sha', 'benchmark://cbench-v1/stringsearch', 'benchmark://cbench-v1/stringsearch2', 'benchmark://cbench-v1/susan', 'benchmark://cbench-v1/tiff2bw', 'benchmark://cbench-v1/tiff2rgba', 'benchmark://cbench-v1/tiffdither', 'benchmark://cbench-v1/tiffmedian']
# runnable_benchmarks = ['benchmark://cbench-v1/bitcount']
results = {}

for benchmark in tqdm(runnable_benchmarks):
    results[benchmark] = {}
    for flag in optimization_flags:
        runtime, inst_count = evaluate_with_flag(benchmark, flag)
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

