from typing import Optional

import numpy as np
import pandas as pd

from ocean_navigation_simulator.problem_factories.FileProblemFactory import FileProblemFactory
from ocean_navigation_simulator.reinforcement_learning.OceanEnv import OceanEnv


class OceanEnvFactory:
    """
    Delivers Env for rllib. It can split up indices s.t. each worker has unique
    training data.
    """

    def __init__(
        self,
        config: dict,
        feature_constructor_config: dict,
        reward_function_config: dict,
        num_workers: Optional[int] = None,
        result_root: Optional[str] = None,
        verbose: Optional[int] = 0,
    ):
        self.config = config
        self.feature_constructor_config = feature_constructor_config
        self.reward_function_config = reward_function_config
        self.num_workers = num_workers
        self.result_root = result_root
        self.verbose = verbose

        self.split_indices()

    def split_indices(self):
        if self.num_workers == 0:
            self.indices = self.num_workers * [None]
        else:
            self.indices = FileProblemFactory.split_indices(
                csv_file=f"{self.config['problem_folder']}problems.csv",
                split=self.num_workers,
                leave=self.config["validation_length"],
            )

    def __call__(self, env_config):
        # env_config: env_config.num_workers, env_config.worker_index, env_config.vector_index, env_config.remote
        return OceanEnv(
            config=self.config,
            feature_constructor_config=self.feature_constructor_config,
            reward_function_config=self.reward_function_config,
            indices=self.indices[env_config.worker_index % self.num_workers]
            if self.num_workers > 0
            else None,
            worker_index=env_config.worker_index,
            result_folder=f"{self.result_root}worker{env_config.worker_index}",
            verbose=self.verbose,
        )
