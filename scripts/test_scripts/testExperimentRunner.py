from ocean_navigation_simulator.ocean_observer.ExperimentRunner import ExperimentRunner


def main():
    """
    Run an experiment
    """
    # np.random.seed(0)
    exp = ExperimentRunner("config_GP_for_failed_case")
    results = exp.run_all_problems()
    print("final results:", results)

if __name__ == "__main__":
    main()


#%%
