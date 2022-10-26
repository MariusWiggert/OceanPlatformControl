from typing import Optional

from ocean_navigation_simulator.reinforcement_learning.env.OceanEnv import (
    OceanEnv,
)
from ocean_navigation_simulator.reinforcement_learning.missions.FileProblemFactory import (
    FileProblemFactory,
)


class OceanEnvFactory:
    """
    Delivers Env for rllib. It can split up indices s.t. each worker has unique training data.
    """

    def __init__(
        self,
        config: dict,
        feature_constructor_config: dict,
        reward_function_config: dict,
        train_workers: Optional[int] = None,
        eval_workers: Optional[int] = None,
        result_root: Optional[str] = None,
        verbose: Optional[int] = 0,
    ):
        self.config = config
        self.feature_constructor_config = feature_constructor_config
        self.reward_function_config = reward_function_config
        self.train_workers = train_workers
        self.eval_workers = eval_workers
        self.result_root = result_root
        self.verbose = verbose

        if self.train_workers > 0:
            self.train_indices = FileProblemFactory.split_indices(
                csv_file=f"{self.config['train_missions']['folder']}problems.csv",
                filter=self.config["train_missions"]["filter"],
                split=self.train_workers,
            )
        if self.eval_workers > 0:
            self.eval_indices = FileProblemFactory.split_indices(
                csv_file=f"{self.config['eval_missions']['folder']}problems.csv",
                filter=self.config["eval_missions"]["filter"],
                split=self.eval_workers,
            )

    def __call__(self, env_config):
        # env_config: env_config.num_workers, env_config.worker_index, env_config.vector_index, env_config.remote
        if env_config.get("evaluation", False) and self.eval_workers > 0:
            indices = self.eval_indices[env_config.worker_index % self.eval_workers]
        elif not env_config.get("evaluation", False) and self.train_workers > 0:
            indices = self.train_indices[env_config.worker_index % self.train_workers]
        else:
            indices = None

        if env_config.get("evaluation", False):
            missions = self.config["eval_missions"]
        else:
            missions = self.config["train_missions"]

        return OceanEnv(
            config=self.config,
            feature_constructor_config=self.feature_constructor_config,
            reward_function_config=self.reward_function_config,
            missions=missions,
            indices=indices,
            evaluation=env_config.get("evaluation", False),
            worker_index=env_config.worker_index,
            result_folder=f"{self.result_root}worker{env_config.worker_index}",
            verbose=self.verbose,
        )
