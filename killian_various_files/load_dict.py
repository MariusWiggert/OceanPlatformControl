import pickle

path = "ablation_study/saved_dictionary_testing.pkl"
with open(path, 'rb') as f:
    full_dict = pickle.load(f)

print(full_dict.keys())

for tile in ["big", "medium"]:
    ls_GP = ['& \multicolumn{2}{l|}{ GP} &']

    # for m in ["r2", "rmse_improved", "rmse_initial", "rmse_ratio", "evm_improved", "evm_initial", "evm_ratio",
    #           "ratio_per_tile"]:
    for m in ['r2', 'rmse_improved', 'evm_improved', 'ratio_per_tile']:
        # GP result
        m += "_GP_validation"
        s = " "
        for i in range(4):
            s += f' {full_dict[f"{tile}_tile_big_ds"][m + f"_{i}"]:.3f},'
        ls_GP.append(s[:-1] + " &")

    print(("".join(ls_GP))[:-1] + "\\\\")

    # All combinations
    for i, ds in enumerate(["big", "medium", "small"]):
        l = "\\arrayrulecolor[rgb]{0.753,0.753,0.753}\\cline{2-7}" if i == 0 else "\\cline{2-11}\n"
        legend = ds[0].capitalize() + ds[1:]
        ls_NN = [f"{l} & {legend} ds  & 3D-Unet &"]
        ls_gp_NN = [f"{l} &                & GP-3D-Unet &"]
        for m in ["r2", "rmse_improved", "evm_improved", "ratio_per_tile"]:
            # for m in ["r2", "rmse_improved", "rmse_initial", "rmse_ratio", "evm_improved", "evm_initial", "evm_ratio",
            #          "ratio_per_tile"]:
            m += "_validation"
            # print(f"{tile}_tile_{ds}_ds")
            s = " "
            t = " "
            for i in range(4):
                s += f'{full_dict[f"{tile}_tile_{ds}_ds_without_gp"][m + f"_{i}"]:.3f},'
                t += f' {full_dict[f"{tile}_tile_{ds}_ds"][m + f"_{i}"]:.3f},'

            ls_NN.append(s[:-1] + " &")
            ls_gp_NN.append(t[:-1] + " &")
        # ls_NN.append(f' {full_dict[f"{tile}_tile_{ds}_ds_without_gp"][m]:.3f} &')
        # ls_gp_NN.append(f' {full_dict[f"{tile}_tile_{ds}_ds"][m]:.3f} &')
        # print('---')
        print((''.join(ls_NN))[:-1] + "\\\\")
        print(("".join(ls_gp_NN))[:-1] + "\\\\")
    print("\n\n")
