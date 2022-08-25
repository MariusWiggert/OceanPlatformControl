from ocean_navigation_simulator.generative_error_model.models.SimplexNoiseModel import SimplexNoiseModel, HarmonicParameters
from ocean_navigation_simulator.generative_error_model.models.GenerativeModel import GenerativeModel
from ocean_navigation_simulator.generative_error_model.utils import convert_degree_to_km
from ocean_navigation_simulator.generative_error_model.Problem import Problem
from ocean_navigation_simulator.utils import units

from typing import List, Dict
import numpy as np
import xarray as xr
import datetime


class OceanCurrentNoiseField(GenerativeModel):
    """Uses noise model to construct a field of noise values."""

    def __init__(self, harmonic_params: List[Dict[str, List[float]]]):
        u_comp_harmonics = [HarmonicParameters(*harmonic) for harmonic in harmonic_params["U_COMP"]]
        v_comp_harmonics = [HarmonicParameters(*harmonic) for harmonic in harmonic_params["V_COMP"]]
        self.model = SimplexNoiseModel(u_comp_harmonics, v_comp_harmonics)

    def reset(self, rng: np.random.default_rng) -> None:
        """Initializes the simplex noise with a new random number generator."""
        self.model.reset(rng)

    def get_noise(self, problem: Problem) -> xr.Dataset:
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
        t_locs = timedelta_range_hours(datetime.timedelta(hours=0), t_range[1] - t_range[0])

        noise = np.zeros((len(lon_locs), len(lat_locs), len(t_locs), 2))

        # TODO: improve on prototype with nested for loops.
        for i, lon in enumerate(lon_locs):
            for j, lat in enumerate(lat_locs):
                for k, elapsed_time in enumerate(t_locs):
                    # need relative degrees, convert to km and feed into noise model
                    x, y = convert_degree_to_km(lon-lon_range[0], lat-lat_range[0])
                    x_km = units.Distance(km=x)
                    y_km = units.Distance(km=y)
                    point_noise = self.model.get_noise(x_km, y_km, elapsed_time)
                    noise[i, j, k, :] = np.array([float(point_noise.u.meters_per_second),
                                                  float(point_noise.v.meters_per_second)])

        ds = xr.Dataset(
            data_vars=dict(
                u_error=(["lon", "lat", "time"], noise[:, :, :, 0]),
                v_error=(["lon", "lat", "time"], noise[:, :, :, 1]),
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


def test():
    # define the components instead of receiving them from OceanCurrentNoiseField
    harmonic_params = {"U_COMP": [[0.5, 702.5, 1407.3, 245.0], [0.5, 302.5, 1207.3, 187.0]],
                       "V_COMP": [[0.5, 702.5, 1407.3, 245.0], [0.5, 302.5, 1207.3, 187.0]]}

    noise_field = OceanCurrentNoiseField(harmonic_params)
    rng = np.random.default_rng(21)  # try different seeds to see if deterministic
    noise_field.reset(rng)
    lon_range = [20, 22]
    lat_range = [10, 12]
    t_range = [datetime.datetime(2022, 5, 10, 12, 30, 0),
               datetime.datetime(2022, 5, 11, 12, 30, 0)]
    problem = Problem(lon_range, lat_range, t_range)

    noise_field = noise_field.get_noise(problem)
    print(noise_field)


if __name__ == "__main__":
    test()
