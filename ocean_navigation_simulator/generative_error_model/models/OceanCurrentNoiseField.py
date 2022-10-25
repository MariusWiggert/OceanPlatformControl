from ocean_navigation_simulator.generative_error_model.models.SimplexNoiseModel import (
    SimplexNoiseModel,
    HarmonicParameters,
)
from ocean_navigation_simulator.generative_error_model.models.GenerativeModel import GenerativeModel
from ocean_navigation_simulator.utils import units

from typing import List, Dict, Any, Tuple
import xarray as xr
import datetime
import numpy as np


class OceanCurrentNoiseField(GenerativeModel):
    """Uses noise model to construct a field of noise values."""

    def __init__(self, harmonic_params: Dict[str, Any], detrend_statistics: np.ndarray):
        u_comp_harmonics = [HarmonicParameters(*harmonic) for harmonic in harmonic_params["U_COMP"]]
        v_comp_harmonics = [HarmonicParameters(*harmonic) for harmonic in harmonic_params["V_COMP"]]
        self.model = SimplexNoiseModel(u_comp_harmonics, v_comp_harmonics)
        self.detrend_statistics = detrend_statistics

    @staticmethod
    def load_config_from_file(parameter_path):
        parameters = np.load(parameter_path, allow_pickle=True)
        harmonic_params = {
            "U_COMP": parameters.item().get("U_COMP"),
            "V_COMP": parameters.item().get("V_COMP"),
        }
        detrend_stats = np.array(parameters.item().get("detrend_metrics"))
        return OceanCurrentNoiseField(harmonic_params, detrend_stats)

    def reset(self, rng: np.random.default_rng) -> None:
        """Initializes the simplex noise with a new random number generator."""
        self.model.reset(rng)

    def get_noise(
        self, lon_axis: np.ndarray, lat_axis: np.ndarray, t_axis: np.ndarray
    ) -> xr.Dataset:
        """Uses the SimplexNoiseModel to produce a noise field over the specified ranges.
        Assumes the origin is positioned in top left corner at timedelta=0 hrs."""

        # create axis steps
        lon_locs = np.array([lon_val - lon_axis[0] for lon_val in lon_axis])
        lat_locs = np.array([lat_val - lat_axis[0] for lat_val in lat_axis])
        t_locs = np.array([np.timedelta64(time_step - t_axis[0]) for time_step in t_axis])

        # get a row of const lat and varying lon and time.
        # Need to convert from degrees to km first. Can't sample all position at once because degree space
        # is not Euclidean and opensimplex only allows for axis sampling. Therefore, sample one plane of lon
        # and time pairs at a time for which lat is const.
        noise = np.zeros((len(t_locs), len(lat_locs), len(lon_locs), 2), dtype=np.single)
        for i, lat_loc in enumerate(lat_locs):
            # constant latitude for conversion of lon values
            lat_pos = np.full(len(lon_locs), lat_loc)
            x_km, y_km = convert_degree_to_km(lon_locs - lon_axis[0], lat_pos - lat_axis[0])
            y_km = np.array([y_km[0]])

            noise[:, i, :, :] = np.squeeze(self.model.get_noise(x_km, y_km, t_locs), axis=1)

        # reintroduce trends into error
        noise[:, :, :, 0] = (
            noise[:, :, :, 0] * self.detrend_statistics[0, 1] + self.detrend_statistics[0, 0]
        )
        noise[:, :, :, 1] = (
            noise[:, :, :, 1] * self.detrend_statistics[1, 1] + self.detrend_statistics[1, 0]
        )

        return self._create_xarray(noise, lon_axis, lat_axis, t_axis)

    @staticmethod
    def _create_xarray(
        data: np.ndarray, lon_axis: np.ndarray, lat_axis: np.ndarray, t_axis: np.ndarray
    ):

        ds = xr.Dataset(
            data_vars=dict(
                water_u=(["time", "lat", "lon"], data[:, :, :, 0]),
                water_v=(["time", "lat", "lon"], data[:, :, :, 1]),
            ),
            coords=dict(time=t_axis, lat=lat_axis, lon=lon_axis),
            attrs=dict(description="An ocean current error sample over time and space."),
        )
        return ds


def datetime_range_from_timedeltas(start: datetime.datetime, timedeltas: List[datetime.timedelta]):
    return [np.datetime64(start + timedelta) for timedelta in timedeltas]


def timedelta_range_hours(start: datetime.timedelta, end: datetime.timedelta, t_res: float):
    start_hours = timedelta_to_hours(start)
    end_hours = timedelta_to_hours(end)
    return [
        start + datetime.timedelta(hours=n) for n in range(0, int(end_hours - start_hours), t_res)
    ]


def timedelta_to_hours(timedelta: datetime.timedelta):
    return timedelta.days * 24 + timedelta.seconds // 3600


def convert_degree_to_km(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Takes two sets of points, each with a lat and lon degree, and computes the distance between each pair in km.
    Note: e.g. pts1 -> np.array([lon, lat])."""
    # https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    x = lon * 40075.2 * np.cos(lat * np.pi / 360) / 360
    y = lat * (39806.64 / 360)
    return x, y


def test():
    params_path = "tuned_2d_forecast_variogram_area1_[5.0, 1.0]_False_True.npy"
    parameters = np.load(params_path, allow_pickle=True)

    # define the components instead of receiving them from OceanCurrentNoiseField
    harmonic_params = {
        "U_COMP": parameters.item().get("U_COMP"),
        "V_COMP": parameters.item().get("V_COMP"),
    }
    detrend_stats = parameters.item().get("detrend_metrics")

    noise_field = OceanCurrentNoiseField(harmonic_params, np.array(detrend_stats))
    rng = np.random.default_rng(21)  # try different seeds to see if deterministic
    noise_field.reset(rng)

    # define the problem
    lon_interval = [-140, -120]
    lat_interval = [20, 30]
    t_interval = [
        datetime.datetime(2022, 5, 2, 12, 30, 0),
        datetime.datetime(2022, 6, 2, 12, 30, 0),
    ]
    # get the noise
    print(noise_field.get_noise(lon_interval, lat_interval, t_interval))


if __name__ == "__main__":
    test()
