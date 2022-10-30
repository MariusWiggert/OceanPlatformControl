# with open('results_grids.pickle', 'wb') as handle:
#    pickle.dump(results_grids, handle, protocol=pickle.HIGHEST_PROTOCOL)
import pickle

import numpy as np

plot_fixed = True
metric_to_plot = 'r2'
type_set = "testing"
path = f"ablation_study/export_all_results_{type_set}_set/"
if plot_fixed:
    file_025, file_05, file_1 = path + "config_GP_025_12_fixed_export_4.pickle", path + "config_GP_05_12_fixed_export_4.pickle", path + "config_GP_1_12_fixed_export_4.pickle"
else:
    file_025, file_05, file_1 = path + "config_GP_025_12_export_4.pickle", path + "config_GP_05_12_export_4.pickle", path + "config_GP_1_12_export_4.pickle"

with open(file_025, 'rb') as handle_025, open(file_05, 'rb') as handle_05, open(file_1, 'rb') as handle_1:
    results_grids_025 = pickle.load(handle_025)
    results_grids_05 = pickle.load(handle_05)
    results_grids_1 = pickle.load(handle_1)
    all_res = [results_grids_025, results_grids_05, results_grids_1]
    colors = ['r', 'g', 'b']
    names = ["radius: 0.25°", "radius: 0.5°", "radius: 1°"]

    #  Print the results
    print("results for each metric: [mean, std, ci]")
    file_05 = results_grids_05
    r2 = np.array(file_05["r2_all_lags_and_radius"]), "r2"
    rmse_improved = np.array(file_05["rmse_improved_all_lags_and_radius"]), "rmse_improved"
    rmse_initial = np.array(file_05["rmse_initial_all_lags_and_radius"]), "rmse_initial"
    rmse_ratio = np.array(file_05["rmse_ratio_all_lags_and_radius"]), "rmse_ratio"
    vme_improved = np.array(file_05["vme_improved_all_lags_and_radius"]), "vme_improved"
    vme_initial = np.array(file_05["vme_initial_all_lags_and_radius"]), "vme_initial"
    vme_ratio = np.array(file_05["vme_ratio_all_lags_and_radius"]), "vme_ratio"
    rpt = np.array(file_05["ratio_per_tile_all_lags_and_radius"]), "ratio_per_tile"
    metrics = [r2, rmse_improved, rmse_initial, rmse_ratio, vme_improved, vme_initial, vme_ratio, rpt]
    # 0.25 deg -> 3 pts radius->idx 3, 0.5 -> 6pts radius 1 -> 12 pts radius
    res = [["" for _ in range(len(metrics))] for _ in range(3)]  # ((len(metrics), 3))
    metrics_names = [m[1] for m in metrics]
    for i, (name, idx) in enumerate([("0.25deg", 3), ("0.5deg", 6), ("1deg", 12)]):
        for j, (m, name_metric) in enumerate(metrics):
            # take only the values merged up to lag 12
            values = m[:, -1, idx]
            mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
            ci = 1.96 * std / np.sqrt(len(values))
            # print(name + " - " + name_metric, mean, std, ci)
            res[i][j] = f"{mean:.2f}+/_{ci:.2f}"
            print(f"{name}  {mean:.2f}+/_{ci:.2f}")
    # print(metrics_names)
    # print("res", res)
    header = f"textbf{{{metrics_names[0]}}}"
    for m in metrics_names[1:]:
        header += f"  & textbf{{{m}}}"
    print(header)
    for arr in res:
        content = f"{arr[0]}"
        for m in arr[1:]:
            content += f"  & {m}"
        print(content)
    '''
    for key in results_grids_025.keys():
        if key.startswith(metric_to_plot):
            to_plots = [np.array(res[key]).mean(axis=0) for res in all_res]
            stds = [np.array(res[key]).std(axis=0) for res in all_res]
            len_objs = len(all_res[0][key])
            if key.endswith("_all_lags_and_radius"):
                plot_2d = True
                legend = "All lags and radius merged"
                title_2d = f"average r2 for the tilesets - {type_set} set"
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
                # plt.title(name + " - " + legend_2d)
                plt.title(title_2d)
                ax_2d.set_xlabel("radius [degree]")
                ax_2d.set_ylabel("R2")

            ha.set_zlabel(name)
            fake2Dlines = []
            for i, to_plot in enumerate(to_plots):
                x, y = range(to_plot.shape[0]), np.arange(0, to_plot.shape[1] / 12, 1 / 12)
                X, Y = np.meshgrid(range(to_plot.shape[0]),
                                   np.arange(0, to_plot.shape[1] / 12,
                                             1 / 12))  # `plot_surface` expects `x` and `y` data to be 2D
                ha.plot_surface(X.T, Y.T, to_plot, color=colors[i])
                fake2Dlines.append(mpl.lines.Line2D([0], [0], linestyle="none", c=colors[i], marker='o'))
                if plot_2d:
                    ci = 1.96 * stds[i][-1] / np.sqrt(len_objs)
                    ax_2d.plot(y, to_plot[-1], c=colors[i])
                    ax_2d.fill_between(y, (to_plot[-1] - ci), (to_plot[-1] + ci), color=colors[i], alpha=.1)

            ha.legend(fake2Dlines, names, numpoints=1)
            if plot_2d:
                ax_2d.legend(fake2Dlines, names)
                
    '''
    print("over")
