from SimplexNoiseModel import SimplexNoiseModel
from ocean_navigation_simulator.utils import units

from typing import Tuple
import numpy as np
import datetime


class OceanCurrentNoiseField:
    """Uses noise model to construct a field of noise values"""

    def __init__(self):
        self.model = SimplexNoiseModel()


    def reset(self, rng) -> None:
        """Initializes the simplex noise with a new random number generator."""
        self.model.reset(rng)
    

    def get_noise_field(self, x_range: Tuple[int], y_range: Tuple[int],\
        elapsed_time_range: Tuple[datetime.timedelta]) -> None:
        """Uses the SimplexNoiseModel to produce a noise field over the specified ranges.
        Assumes the origin is positioned in top left corner at timedelta=0 hrs."""


        # for now hardcode resolutions
        x_res, y_res, t_res = 1/12,1/12,1
        x_locs = np.arange(x_range[0], x_range[1], x_res)
        y_locs = np.arange(y_range[0], y_range[1])
        t_locs = timedeltarange(elapsed_time_range[0], elapsed_time_range[1])

        noise = np.zeros((len(x_locs), len(y_locs), len(t_locs)), dtype=object)

        # prototype with nested for loops -> still surprisingly fast for realistic resolution.
        for i, x in enumerate(x_locs):
            for j, y in enumerate(y_locs):
                for k, elapsed_time in enumerate(t_locs):
                    x_km = units.Distance(km=x)
                    y_km = units.Distance(km=y)
                    noise[i,j,k] = self.model.get_noise(x_km, y_km, elapsed_time)
        return noise


def timedeltarange(start:datetime.timedelta, end:datetime.timedelta):
    start_hours = timedelta_to_hours(start)
    end_hours = timedelta_to_hours(end)
    return [start + datetime.timedelta(hours=n) for n in range(int(end_hours-start_hours))]

def timedelta_to_hours(timedelta: datetime.timedelta):
    return timedelta.days*24 + timedelta.seconds//3600


if __name__ == "__main__":
    noise = OceanCurrentNoiseField()
    rng = np.random.default_rng()
    noise.reset(rng)
    t_delta_range = (datetime.timedelta(days=0), datetime.timedelta(days=9))
    noise_field = noise.get_noise_field((20,40), (10,20), t_delta_range)
    print(noise_field)