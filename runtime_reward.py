from typing import Callable, Iterable, List, Optional
import numpy as np
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import CompilerEnvWrapper



class RuntimeInstCountReward(Reward):
    def __init__(
        self,
        runtime_count: int,
        warmup_count: int,
        estimator: Callable[[Iterable[float]], float],
        default_value: int = 0,
    ):
        super().__init__(
            id="runtime",
            observation_spaces=["Runtime"],
            default_value=default_value,
            min=None,
            max=None,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.runtime_count = runtime_count
        self.warmup_count = warmup_count
        self.starting_runtime: Optional[float] = None
        self.starting_inst_count: Optional[int] = None
        self.previous_runtime: Optional[float] = None
        self.current_benchmark: Optional[str] = None
        self.estimator = estimator

    def reset(self, benchmark, observation_view) -> None:
        # If we are changing the benchmark then check that it is runnable.
        if benchmark != self.current_benchmark:
            if not observation_view["IsRunnable"]:
                raise ValueError(f"Benchmark is not runnable: {benchmark}")
            self.current_benchmark = benchmark
            self.starting_runtime = None
            self.starting_inst_count = None

        # Compute initial runtime if required, else use previously computed
        # value.
        if self.starting_runtime is None:
            self.starting_runtime = self.estimator(observation_view["Runtime"])
        if self.starting_inst_count is None:
            self.starting_inst_count = observation_view['IrInstructionCount']

        self.previous_runtime = self.starting_runtime
        self.previous_inst_count = self.starting_inst_count

    def update(
        self,
        actions: List[ActionType],
        observations: List[ObservationType],
        observation_view,
    ) -> float:
        
        alpha = 0.5

        del actions  # unused
        runtimes = observations[0]
        if len(runtimes) != self.runtime_count:
            raise ValueError(
                f"Expected {self.runtime_count} runtimes but received {len(runtimes)}"
            )
        runtime = self.estimator(runtimes)
        inst_count = observation_view['IrInstructionCount']

        runtime_reward = self.previous_runtime - runtime
        inst_count_reward = self.previous_inst_count - inst_count

        self.previous_runtime = runtime
        self.previous_inst_count = inst_count
        reward = alpha*runtime_reward + (1-alpha)*inst_count_reward

        return reward



class RuntimeInstCountRewardWrapper(CompilerEnvWrapper):
    """LLVM wrapper that uses a point estimate of program runtime as reward.

    This class wraps an LLVM environment and registers a new runtime reward
    space. Runtime is estimated from one or more runtime measurements, after
    optionally running one or more warmup runs. At each step, reward is the
    change in runtime estimate from the runtime estimate at the previous step.
    """

    def __init__(
        self,
        env: LlvmEnv,
        runtime_count: int = 30,
        warmup_count: int = 0,
        estimator: Callable[[Iterable[float]], float] = np.median,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param runtime_count: The number of times to execute the binary when
            estimating the runtime.

        :param warmup_count: The number of warmup runs of the binary to perform
            before measuring the runtime.

        :param estimator: A function that takes a list of runtime measurements
            and produces a point estimate.
        """
        super().__init__(env)

        self.env.unwrapped.reward.add_space(
            RuntimeInstCountReward(
                runtime_count=runtime_count,
                warmup_count=warmup_count,
                estimator=estimator,
            )
        )
        self.env.unwrapped.reward_space = "runtime"

        self.env.unwrapped.runtime_observation_count = runtime_count
        self.env.unwrapped.runtime_warmup_runs_count = warmup_count

    def fork(self) -> "RuntimeInstCountRewardWrapper":
        fkd = self.env.fork()
        # Remove the original "runtime" space so that we that new
        # RuntimePointEstimateReward wrapper instance does not attempt to
        # redefine, raising a warning.
        del fkd.unwrapped.reward.spaces["runtime"]
        return RuntimeInstCountRewardWrapper(
            env=fkd,
            runtime_count=self.reward.spaces["runtime"].runtime_count,
            warmup_count=self.reward.spaces["runtime"].warmup_count,
            estimator=self.reward.spaces["runtime"].estimator,
        )
