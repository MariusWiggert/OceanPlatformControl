from ocean_navigation_simulator.generative_error_model.models.SimplexNoiseModel import SimplexNoiseModel, HarmonicParameters
from ocean_navigation_simulator.generative_error_model.models.GenerativeModel import GenerativeModel
from ocean_navigation_simulator.generative_error_model.utils import convert_degree_to_km
from ocean_navigation_simulator.generative_error_model.Problem import Problem
from ocean_navigation_simulator.generative_error_model.utils import timer, get_path_to_project, load_config
from ocean_navigation_simulator.utils import units

from typing import List, Dict, Any
import xarray as xr
import datetime
import numpy as np
import os


class OceanCurrentNoiseField(GenerativeModel):
    """Uses noise model to construct a field of noise values."""

    def __init__(self, harmonic_params: Dict[str, Any], detrend_statistics: np.ndarray):
        u_comp_harmonics = [HarmonicParameters(*harmonic) for harmonic in harmonic_params["U_COMP"]]
        v_comp_harmonics = [HarmonicParameters(*harmonic) for harmonic in harmonic_params["V_COMP"]]
        self.model = SimplexNoiseModel(u_comp_harmonics, v_comp_harmonics)
        self.detrend_statistics = detrend_statistics

    def reset(self, rng: np.random.default_rng) -> None:
        """Initializes the simplex noise with a new random number generator."""
        self.model.reset(rng)

    def get_noise(self, problem: Problem) -> xr.Dataset:
        """Uses the SimplexNoiseModel to produce a noise field over the specified ranges.
        Assumes the origin is positioned in top left corner at timedelta=0 hrs."""

        # TODO: user-defined resolutions, since they vary between HYCOM and Copernicus.
        lon_res, lat_res, t_res = 1 / 12, 1 / 12, 1
        lon_range, lat_range, t_range = problem.lon_range, problem.lat_range, problem.t_range

        # drop timezome from datetime
        t_range[0] = t_range[0].replace(tzinfo=None)
        t_range[1] = t_range[1].replace(tzinfo=None)

        # create axis steps
        lon_locs = np.arange(lon_range[0], lon_range[1], lon_res)
        lat_locs = np.arange(lat_range[0], lat_range[1], lat_res)
        t_locs = timedelta_range_hours(datetime.timedelta(hours=0), t_range[1] - t_range[0])

        noise = np.zeros((len(lon_locs), len(lat_locs), len(t_locs), 2))

        # TODO: improve on prototype with nested for loops.
        for i, lon in enumerate(lon_locs):
            for j, lat in enumerate(lat_locs):
                for k, elapsed_time in enumerate(t_locs):
                    # need degrees relative to start of locs, convert to km and feed into noise model
                    x, y = convert_degree_to_km(lon - lon_range[0], lat - lat_range[0])
                    x_km = units.Distance(km=x)
                    y_km = units.Distance(km=y)
                    point_noise = self.model.get_noise(x_km, y_km, elapsed_time)
                    noise[i, j, k, :] = np.array([float(point_noise.u.meters_per_second),
                                                  float(point_noise.v.meters_per_second)])

        # reintroduce trends into error
        noise[:, :, :, 0] = noise[:, :, :, 0] * self.detrend_statistics[0, 1] + self.detrend_statistics[0, 0]
        noise[:, :, :, 1] = noise[:, :, :, 1] * self.detrend_statistics[1, 1] + self.detrend_statistics[1, 0]

        return self._create_xarray(noise, lon_locs, lat_locs, t_locs, t_range)

    def get_noise_vec(self, problem: Problem) -> xr.Dataset:
        """Uses the SimplexNoiseModel to produce a noise field over the specified ranges.
        Assumes the origin is positioned in top left corner at timedelta=0 hrs."""

        # TODO: user-defined resolutions, since they vary between HYCOM and Copernicus.
        lon_res, lat_res, t_res = 1/12, 1/12, 1
        lon_range, lat_range, t_range = problem.lon_range, problem.lat_range, problem.t_range

        # drop timezome from datetime
        t_range[0] = t_range[0].replace(tzinfo=None)
        t_range[1] = t_range[1].replace(tzinfo=None)

        # create axis steps
        lon_locs = np.arange(lon_range[0], lon_range[1], lon_res)
        lat_locs = np.arange(lat_range[0], lat_range[1], lat_res)
        t_locs = np.array(timedelta_range_hours(datetime.timedelta(hours=0), t_range[1] - t_range[0]))
        # convert to numpy otherwise numba throws a fit! # possible to do timedelta.to_timedelta64()
        t_locs = np.array([np.timedelta64(timedelta) for timedelta in t_locs])

        # get a row of const lat and varying lon and time.
        # Need to convert from degrees to km first. Cant sample all position at once because degree space
        # is not Euclidean and opensimplex only allows for axis sampling. Therefore, sample one plane of lon
        # and time pairs at a time for which lat is const.
        noise = np.zeros((len(lon_locs), len(lat_locs), len(t_locs), 2))
        for i, lat_loc in enumerate(lat_locs):
            # constant latitude for conversion of lon values
            lat_pos = np.full(len(lon_locs), lat_loc)
            x_km, y_km = convert_degree_to_km(lon_locs - lon_range[0], lat_pos - lat_range[0])
            y_km = np.array([y_km[0]])

            noise[:, i, :, :] = np.squeeze(self.model.get_noise_vec(x_km, y_km, t_locs), axis=1)

        # reintroduce trends into error
        noise = noise.reshape(len(lon_locs), len(lat_locs), len(t_locs), 2)

        # print(f"u metrics: {noise[:, :, :, 0].mean()}, {np.sqrt(noise[:, :, :, 0].var())}")
        # print(f"v metrics: {noise[:, :, :, 1].mean()}, {np.sqrt(noise[:, :, :, 1].var())}")
        # # what we want:
        # print(self.detrend_statistics[0, 0], self.detrend_statistics[0, 1])
        # print(self.detrend_statistics[1, 0], self.detrend_statistics[1, 1])

        noise[:, :, :, 0] = noise[:, :, :, 0] * self.detrend_statistics[0, 1] + self.detrend_statistics[0, 0]
        noise[:, :, :, 1] = noise[:, :, :, 1] * self.detrend_statistics[1, 1] + self.detrend_statistics[1, 0]

        # print(f"u metrics: {noise[:, :, :, 0].mean()}, {np.sqrt(noise[:, :, :, 0].var())}")
        # print(f"v metrics: {noise[:, :, :, 1].mean()}, {np.sqrt(noise[:, :, :, 1].var())}")

        # need to convert time to list
        t_locs = t_locs.tolist()

        return self._create_xarray(noise, lon_locs, lat_locs, t_locs, t_range)

    def _create_xarray(self, data: np.ndarray, lon_locs: np.ndarray, lat_locs: np.ndarray,
                       t_locs: List[datetime.timedelta], t_range: List[datetime.datetime]):

        ds = xr.Dataset(
            data_vars=dict(
                u_error=(["lon", "lat", "time"], data[:, :, :, 0]),
                v_error=(["lon", "lat", "time"], data[:, :, :, 1]),
            ),
            coords=dict(
                lon=lon_locs,
                lat=lat_locs,
                time=datetime_range_from_timedeltas(t_range[0], t_locs)
            ),
            attrs=dict(description="An ocean current error sample over time and space.")
        )
        return ds


def datetime_range_from_timedeltas(start: datetime.datetime, timedeltas: List[datetime.timedelta]):
    return [start + timedelta for timedelta in timedeltas]


def timedelta_range_hours(start: datetime.timedelta, end: datetime.timedelta):
    start_hours = timedelta_to_hours(start)
    end_hours = timedelta_to_hours(end)
    return [start + datetime.timedelta(hours=n) for n in range(int(end_hours-start_hours))]


def timedelta_to_hours(timedelta: datetime.timedelta):
    return timedelta.days*24 + timedelta.seconds//3600


@timer
def test():
    config = load_config("config_buoy_data.yaml")
    project_dir = get_path_to_project(os.getcwd())

    # define the components instead of receiving them from OceanCurrentNoiseField
    parameters_file = os.path.join(project_dir, config["data_dir"], config["model"]["simplex_noise"]["area1"])
    parameters = np.load(parameters_file, allow_pickle=True)
    harmonic_params = {"U_COMP": parameters.item().get("U_COMP"),
                       "V_COMP": parameters.item().get("V_COMP")}
    detrend_stats = parameters.item().get("detrend_metrics")

    noise_field = OceanCurrentNoiseField(harmonic_params, np.array(detrend_stats))
    rng = np.random.default_rng(21)  # try different seeds to see if deterministic
    noise_field.reset(rng)

    # define the problem
    lon_range = [-140, -120]
    lat_range = [20, 30]
    t_range = [datetime.datetime(2022, 4, 21, 12, 30, 0),
               datetime.datetime(2022, 5, 21, 12, 30, 0)]
    problem = Problem(lon_range, lat_range, t_range)

    # get the noise
    noise_field = noise_field.get_noise_vec(problem)
    print(noise_field)
    noise_field.to_netcdf("~/Downloads/plots/forecast_validation/vec_sample_noise.nc")


if __name__ == "__main__":
    test()
