import numpy as np
import xarray as xr

# Constants we use for Kelp (San Francisco Bay Experiment)
# temperature
T_opt = 12  # celsius
T_max = 20  # celsius
T_min = 5  # celsius

# Constants for a specific seaweed species (tropical species)
# TODO: which units are all of them in?
# temperature
# T_opt = 20  # celsius
# T_max = 25  # celsius
# T_min = 0  # celsius
# nutrients
K_N = 2
K_P = 0.1
# irradiance
k_w = 0.04
Z_m = 5  # assume we are at a constant depth 5
I_opt = 180
# gross growth rate
u_max = 0.2
# respiration rate
R_max20 = 1.5 / 100
r = 1.047
# loss rate = erosion of biomass
R_erosion = 0.01

# TODO: Function description, typing, specifically which units the inputs are in.


def irradianceFactor(I_S):
    # Compute Irradiance Factor
    I_ma = I_S * np.e ** (-k_w * Z_m)
    f_Ima = I_ma / I_opt * np.e ** (1 - I_ma / I_opt)
    return f_Ima


def temperatureFactor(T_W):
    # Compute temperature factor
    T_x = np.where(T_W <= T_opt, T_min, T_max)
    X_T = (T_W - T_opt) / (T_x - T_opt)
    f_T = np.e ** (-2.3 * (X_T**2))
    return f_T


def nutrientFactor(NO_3, PO_4):
    # Compute factor of Nitrate and Phosphate
    # nutrients
    f_N = NO_3 / (K_N + NO_3)
    f_P = PO_4 / (K_P + PO_4)
    f_NP = np.minimum(f_N, f_P)
    return f_NP


def compute_R_resp(T_W):
    """Computes respiration rate per day."""
    R_resp = R_max20 * r ** (T_W - 20)
    return R_resp


def compute_R_growth_without_irradiance(T_W, NO_3, PO_4):  # for one point or arrays
    # TODO: Function description, typing, specifically which units the inputs are in.
    # gross growth rate
    R_growth_wo_irradiance = u_max * temperatureFactor(T_W) * nutrientFactor(NO_3, PO_4)
    return R_growth_wo_irradiance


def compute_NGR(T_W, NO_3, PO_4, I_S):  # for one point
    R_growth = u_max * temperatureFactor(T_W) * nutrientFactor(NO_3, PO_4) * irradianceFactor(I_S)
    NGR = R_growth - compute_R_resp(T_W)
    return NGR


def compute_LR(biomass):
    return biomass * R_erosion


def compute_dBiomass_dT(T_W, NO_3, PO_4, I_S, biomass):
    return biomass * (compute_NGR(T_W, NO_3, PO_4, I_S) - compute_LR(biomass))


if __name__ == "__main__":
    f = xr.open_dataset("./data2021_monthly_nutrients_and_temp.nc")
    f.assign(R_growth=compute_R_growth_without_irradiance(f["Temperature"], f["no3"], f["po4"]))
