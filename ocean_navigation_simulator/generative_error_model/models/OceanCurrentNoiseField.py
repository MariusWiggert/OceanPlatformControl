from ocean_navigation_simulator.generative_error_model.models.SimplexNoiseModel import SimplexNoiseModel, HarmonicParameters
from ocean_navigation_simulator.generative_error_model.models.GenerativeModel import GenerativeModel
from ocean_navigation_simulator.generative_error_model.utils import convert_degree_to_km
from ocean_navigation_simulator.generative_error_model.utils import timer

from typing import List, Dict, Any, Optional
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
        self.time_origin = None

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
        self.time_origin = None

    def get_noise_from_ranges(self,
                              x_interval: List[float],
                              y_interval: List[float],
                              t_interval: List[datetime.datetime],
                              spatial_resolution: Optional[float] = None,
                              temporal_resolution: Optional[float] = None
                  ) -> xr.Dataset:
        """Uses the SimplexNoiseModel to produce a noise field over the specified ranges.
        Assumes the origin is positioned in top left corner at timedelta=0 hrs.

        Parameters:
            spatial_resolution - in degrees
            temporal_resolution - in hours
        """

        if temporal_resolution is None:
            lon_res, lat_res, = 1/12, 1/12
        else:
            lon_res, lat_res = spatial_resolution, spatial_resolution

        if spatial_resolution is None:
            t_res = 1
        else:
            t_res = temporal_resolution
        lon_range, lat_range, t_range = x_interval, y_interval, t_interval

        # Drop timezome from datetime
        t_range[0] = t_range[0].replace(tzinfo=None)
        t_range[1] = t_range[1].replace(tzinfo=None)

        # Create axis locations. Time is taken relative to first timestamp, spatial axis are global.
        lon_locs = np.arange(lon_range[0], lon_range[1] + lon_res/2, lon_res)
        lat_locs = np.arange(lat_range[0], lat_range[1] + lat_res/2, lat_res)
        if self.time_origin is None:
            t_locs = np.array(timedelta_range_hours(datetime.timedelta(hours=0),
                                                    t_range[1] - t_range[0] + datetime.timedelta(hours=t_res),
                                                    t_res))
        else:
            t_locs = np.array(timedelta_range_hours(t_range[0] - self.time_origin,
                                                    t_range[1] - t_range[0] + datetime.timedelta(hours=t_res),
                                                    t_res))
        t_locs = np.array([np.timedelta64(timedelta) for timedelta in t_locs])

        # set time_origin in case perform inference for same seed
        self.time_origin = t_range[0]

        return self.get_noise(lon_locs, lat_locs, t_locs, t_range[0])

    def get_noise_from_axes(
        self, lon_axis: np.ndarray, lat_axis: np.ndarray, t_axis: np.ndarray
    ) -> xr.Dataset:
        """Uses the SimplexNoiseModel to produce a noise field over the specified axes arrays."""

        # make time relative to first time step.
        if self.time_origin is None:
            t_locs = np.array([np.timedelta64(time_step - t_axis[0]) for time_step in t_axis])
        else:
            t_locs = np.array([np.timedelta64(time_step - t_axis[0]) for time_step in t_axis - self.time_origin])

        # set time_origin in case perform inference for same seed
        self.time_origin = t_axis[0]

        return self.get_noise(lon_axis, lat_axis, t_locs, t_axis[0])

    def get_noise(self,
                  lon_locs: np.ndarray,
                  lat_locs: np.ndarray,
                  t_locs: np.ndarray,
                  t_origin: datetime.datetime):

        """This method iteratively gets a slice of constant lat and varying lon and time with every iteration.
        Cannot sample all positions at once because degree-space is not Euclidean and OpenSimplex only allows
        for axis sampling.
        Therefore, the method samples one (2D) slice of lon and time pairs at a time for which lat is constant.
        Convert from degrees to km first before passing into get_noise method."""

        noise = np.zeros((len(t_locs), len(lat_locs), len(lon_locs), 2))
        for i, lat_loc in enumerate(lat_locs):
            # constant latitude for conversion of lon values
            lat_pos = np.full(len(lon_locs), lat_loc)
            x_km, y_km = convert_degree_to_km(lon_locs, lat_pos)
            y_km = np.array([y_km[0]])

            # return noise of shape [time, lat, lon, current_component]
            noise[:, i, :, :] = np.squeeze(self.model.get_noise(x_km, y_km, t_locs), axis=1)

        # # mean and std before statistics reintroduction
        # print(f"u metrics: {noise[:, :, :, 0].mean()}, {np.sqrt(noise[:, :, :, 0].var())}")
        # print(f"v metrics: {noise[:, :, :, 1].mean()}, {np.sqrt(noise[:, :, :, 1].var())}")
        # # what we want:
        # print(self.detrend_statistics[0, 0], self.detrend_statistics[0, 1])
        # print(self.detrend_statistics[1, 0], self.detrend_statistics[1, 1])

        # reintroduce trends into error
        noise[:, :, :, 0] = (
                noise[:, :, :, 0] * self.detrend_statistics[0, 1] + self.detrend_statistics[0, 0]
        )
        noise[:, :, :, 1] = (
                noise[:, :, :, 1] * self.detrend_statistics[1, 1] + self.detrend_statistics[1, 0]
        )

        # print(f"u metrics: {noise[:, :, :, 0].mean()}, {np.sqrt(noise[:, :, :, 0].var())}")
        # print(f"v metrics: {noise[:, :, :, 1].mean()}, {np.sqrt(noise[:, :, :, 1].var())}")

        return self._create_xarray(noise, lon_locs, lat_locs, t_locs, t_origin)

    @staticmethod
    def _create_xarray(
        data: np.ndarray, lon_axis: np.ndarray, lat_axis: np.ndarray, t_axis: np.ndarray, t_start: datetime.datetime
    ) -> xr.Dataset:

        ds = xr.Dataset(
            data_vars=dict(
                water_u=(["time", "lat", "lon"], data[:, :, :, 0]),
                water_v=(["time", "lat", "lon"], data[:, :, :, 1]),
            ),
            coords=dict(
                time=datetime_range_from_timedeltas(t_start, t_axis.tolist()),
                lat=lat_axis,
                lon=lon_axis
            ),
            attrs=dict(description="An ocean current error sample over time and space."),
        )
        return ds


def datetime_range_from_timedeltas(start: datetime.datetime, timedeltas: List[datetime.timedelta]):
    return [start + timedelta for timedelta in timedeltas]


def timedelta_range_hours(start: datetime.timedelta, end: datetime.timedelta, t_res: int):
    start_hours = timedelta_to_hours(start)
    end_hours = timedelta_to_hours(end)
    return [start + datetime.timedelta(hours=n) for n in range(0, int(end_hours-start_hours), t_res)]


def timedelta_to_hours(timedelta: datetime.timedelta):
    return timedelta.days*24 + timedelta.seconds//3600


@timer
def main():
    # params_path = "data/drifter_data/variogram_params/tuned_2d_forecast_variogram_area1_[5.0, 1.0]_False_True.npy"
    params_path = "data/drifter_data/variogram_params/tuned_2d_forecast_variogram_area1_edited2.npy"

    print(np.load(params_path, allow_pickle=True))

    noise_field = OceanCurrentNoiseField.load_config_from_file(params_path)
    rng = np.random.default_rng(123)  # change seed to get different universe of noise
    noise_field.reset(rng)

    # define the problem
    lon_range = [-140, -120]
    lat_range = [20, 30]
    t_range = [datetime.datetime(2022, 5, 2, 12, 30, 0),
               datetime.datetime(2022, 6, 2, 12, 30, 0)]

    # get the noise
    noise = noise_field.get_noise_from_ranges(lon_range, lat_range, t_range)
    print(noise)
    noise.to_netcdf("/home/jonas/Downloads/test_noise.nc")


if __name__ == "__main__":
    main()
