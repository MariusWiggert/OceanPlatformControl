import numpy as np
from statsmodels.stats.proportion import proportions_ztest

#%% Hycom-Hycom
N_success_multi = 1123
N_success_straight = 977
N_total = 1151
count = np.array([N_success_multi, N_success_straight])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative="larger")
# print(pval)
#%% Hycom-Copernicus
N_success_multi = 658  # 689
N_success_straight = 652
N_total = 837
count = np.array([N_success_multi, N_success_straight])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative="larger")
# print(pval)

#%% Stranding

N_total = 1146

# %%
# switch vs naive
N_stranding_switch = 29
N_stranding_hj_naive = 54
count = np.array([N_stranding_switch, N_stranding_hj_naive])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative="smaller")
print(pval)

# Obs vs naive
N_stranding_hj_obs = 11
N_stranding_hj_naive = 54
count = np.array([N_stranding_hj_obs, N_stranding_hj_naive])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative="smaller")
print(pval)

# switch_obs
N_stranding_switch = 15
N_stranding_hj_naive = 54
count = np.array([N_stranding_switch, N_stranding_hj_naive])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative="smaller")
print(pval)

# obs disturbance
N_stranding_switch = 22
N_stranding_hj_naive = 54
count = np.array([N_stranding_switch, N_stranding_hj_naive])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative="smaller")
print(pval)

# %%
