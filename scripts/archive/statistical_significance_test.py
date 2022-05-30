import numpy as np
from statsmodels.stats.proportion import proportions_ztest

#%% Hycom-Hycom
N_success_multi = 1123
N_success_straight = 977
N_total = 1151
count = np.array([N_success_multi, N_success_straight])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative='larger')
print(pval)
#%% Hycom-Copernicus
N_success_multi = 658#689
N_success_straight = 652
N_total = 837
count = np.array([N_success_multi, N_success_straight])
nobs = np.array([N_total, N_total])
stat, pval = proportions_ztest(count, nobs, alternative='larger')
print(pval)