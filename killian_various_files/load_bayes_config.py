# from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

path_file_save_checkpoint = f"/Users/fedosha/Desktop/output_test/backup_full_4/checkpoints/GP_grid_025_12.pkl"

search_space_bayes = {
    "sigma_exp": (0.00001, 10),
    "latitude": (1e-2, 5),
    "longitude": (1e-2, 5),  # tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
    "time": (7200, 43200)
}
research_state = 1
random_search_space = 55

bayesopt = BayesOptSearch(space=search_space_bayes,
                          metric="r2", mode="max",
                          random_state=research_state,
                          random_search_steps=random_search_space)
if True:
    bayesopt.restore(path_file_save_checkpoint)
    print("loaded bayes config: ", path_file_save_checkpoint)
