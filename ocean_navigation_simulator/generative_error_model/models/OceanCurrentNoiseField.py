from SimplexNoiseModel import SimplexNoiseModel
from ocean_navigation_simulator.utils import units

from typing import Tuple
import numpy as np
import xarray as xr
import datetime


class OceanCurrentNoiseField:
    """Uses noise model to construct a field of noise values"""

    def __init__(self):
        self.model = SimplexNoiseModel()


    def reset(self, rng) -> None:
        """Initializes the simplex noise with a new random number generator."""
        self.model.reset(rng)
    

    def get_noise_field(self, x_range: Tuple[int], y_range: Tuple[int],\
        elapsed_time_range: Tuple[datetime.timedelta]) -> xr.Dataset:
        """Uses the SimplexNoiseModel to produce a noise field over the specified ranges.
        Assumes the origin is positioned in top left corner at timedelta=0 hrs."""

        # TODO: user-defined resolutions, since they vary between HYCOM and Copernicus
        x_res, y_res, t_res = 1/12,1/12,1
        x_locs = np.arange(x_range[0], x_range[1], x_res)
        y_locs = np.arange(y_range[0], y_range[1], y_res)
        t_locs = timedeltarange(elapsed_time_range[0], elapsed_time_range[1])

        noise = np.zeros((len(x_locs), len(y_locs), len(t_locs), 2), dtype=object)

        # TODO: improve on prototype with nested for loops -> still surprisingly fast for realistic resolution...
        for i, x in enumerate(x_locs):
            for j, y in enumerate(y_locs):
                for k, elapsed_time in enumerate(t_locs):
                    x_km = units.Distance(km=x)
                    y_km = units.Distance(km=y)
                    point_noise = self.model.get_noise(x_km, y_km, elapsed_time)
                    noise[i,j,k,:] = np.array([point_noise.u.meters_per_second, point_noise.v.meters_per_second])

        ds = xr.Dataset(
            data_vars=dict(
                u_error=(["x", "y", "time"], noise[:,:,:,0]),
                v_error=(["x", "y", "time"], noise[:,:,:,0]),
            ),
            coords=dict(
                x=x_locs,
                y=y_locs,
                time=t_locs # TODO: need reference time to align with forecast/hindcast
            ),
            attrs=dict(description="Ocean current error over time and space.")
        )
        return ds


def timedeltarange(start:datetime.timedelta, end:datetime.timedelta):
    start_hours = timedelta_to_hours(start)
    end_hours = timedelta_to_hours(end)
    return [start + datetime.timedelta(hours=n) for n in range(int(end_hours-start_hours))]

def timedelta_to_hours(timedelta: datetime.timedelta):
    return timedelta.days*24 + timedelta.seconds//3600


if __name__ == "__main__":
    # run simple example:
    noise = OceanCurrentNoiseField()
    rng = np.random.default_rng(21) # try different seeds to see if deterministic
    noise.reset(rng)
    t_delta_range = (datetime.timedelta(days=0), datetime.timedelta(days=1))
    noise_field = noise.get_noise_field((20,22), (10,12), t_delta_range)
    print(noise_field["u_error"])

