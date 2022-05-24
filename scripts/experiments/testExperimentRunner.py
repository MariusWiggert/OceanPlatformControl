import numpy as np

from ocean_navigation_simulator.env.ExperimentRunner import ExperimentRunner


def main():
    np.random.seed(0)
    exp = ExperimentRunner("config_real_data_GP")
    results = exp.run_all_problems()
    print("final results:", results)


if __name__ == "__main__":
    main()
