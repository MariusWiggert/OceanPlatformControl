# with open('results_grids.pickle', 'wb') as handle:
#    pickle.dump(results_grids, handle, protocol=pickle.HIGHEST_PROTOCOL)
import pickle

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

plot_fixed = True
metric_to_plot = 'r2'
path = "ablation_study/export_all_results_testing_set/"
if plot_fixed:
    file_025, file_05, file_1 = path + "config_GP_025_12_fixed_export_4.pickle", path + "config_GP_05_12_fixed_export_4.pickle", path + "config_GP_1_12_fixed_fixed_export_4.pickle"
else:
    file_025, file_05, file_1 = path + "config_GP_025_12_export_4.pickle", path + "config_GP_05_12_export_4.pickle", path + "config_GP_1_12_export_4.pickle"

with open(file_025, 'rb') as handle_025, open(file_05, 'rb') as handle_05, open(file_1, 'rb') as handle_1:
    results_grids_025 = pickle.load(handle_025)
    results_grids_05 = pickle.load(handle_05)
    results_grids_1 = pickle.load(handle_1)
    all_res = [results_grids_025, results_grids_05, results_grids_1]
    colors = ['r', 'g', 'b']
    names = ["radius: 0.25°", "radius: 0.5°", "radius: 1°"]
    for key in results_grids_025.keys():
        if key.startswith(metric_to_plot):
            to_plots = [np.array(res[key]).mean(axis=0) for res in all_res]
            if key.endswith("_all_lags_and_radius"):
                plot_2d = True
                legend = "All lags and radius merged"
                name = key[:-len("_all_lags_and_radius")]
            else:
                plot_2d = False
                legend = "each lag and radius separated"
                name = key[:-len("per_lag_and_radius")]
            name = name.replace("_", " ")
            hf = plt.figure()  # traditional 2d plot
            ha = hf.add_subplot(111, projection='3d')
            plt.title(name + " - " + legend)
            ha.set_xlabel("lag [h]")
            ha.set_ylabel("radius [degree]")
            if plot_2d:
                ax_2d = plt.figure().gca()
                plt.title(name + " - " + legend)
                ax_2d.set_xlabel("radius [degree]")
                ax_2d.set_ylabel("error")

            ha.set_zlabel(name)
            fake2Dlines = []
            for i, to_plot in enumerate(to_plots):
                x, y = range(to_plot.shape[0]), np.arange(0, to_plot.shape[1] / 12, 1 / 12)
                X, Y = np.meshgrid(range(to_plot.shape[0]),
                                   np.arange(0, to_plot.shape[1] / 12,
                                             1 / 12))  # `plot_surface` expects `x` and `y` data to be 2D
                ha.plot_surface(X.T, Y.T, to_plot, color=colors[i], label="toto")
                fake2Dlines.append(mpl.lines.Line2D([0], [0], linestyle="none", c=colors[i], marker='o'))
                if plot_2d:
                    ax_2d.plot(y, to_plot[-1], c=colors[i])

            ha.legend(fake2Dlines, names, numpoints=1)
            if plot_2d:
                ax_2d.legend(fake2Dlines, names)
    print("over")
