import argparse
import datetime
import math
import os
import random
from typing import Union

import numpy as np
import yaml

from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint, SpatialPoint
from ocean_navigation_simulator.utils.units import Velocity, Distance

number_problems = 100
seed = 30031996
lon_left, lon_right = -96.362841, -84.766062
lat_bottom, lat_top = 23.366677, 28.279219
area_coordinates = [(lon_left, lat_top), (lon_right, lat_top), (lon_left, lat_bottom),
                    (lon_right, lat_bottom)]  # tl, tr, bl, br (lon, lat)
start_date = datetime.datetime(2021, 11, 22, 12, 00, 00)
end_date = datetime.datetime(2021, 11, 28, 12, 00, 00)
duration_simulation = datetime.timedelta(days=3)  # Change to 5 days when we have more files
max_velocity = Velocity(mps=1)
distance_start_end_point = max_velocity * duration_simulation


def generate_end_point(initial_point: SpatialPoint) -> SpatialPoint:
    angle = random.uniform(0, 2 * math.pi)
    new_point = SpatialPoint(Distance(deg=initial_point.lon.deg + math.cos(angle) * distance_start_end_point.deg),
                             Distance(deg=initial_point.lat.deg + math.sin(angle) * distance_start_end_point.deg))
    return new_point if check_point(new_point) else generate_end_point(initial_point)


def generate_point_in_area() -> SpatialPoint:
    long = random.uniform(min(lon_left, lon_right), max(lon_left, lon_right))
    lat = random.uniform(min(lat_bottom, lat_top), max(lat_bottom, lat_top))
    return SpatialPoint(Distance(deg=long), Distance(deg=lat))


def generate_random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    n_sec_per_day = 24 * 60 * 60
    int_delta = (delta.days * n_sec_per_day) + delta.seconds
    random_second = random.randrange(int_delta)
    random_second = (random_second // n_sec_per_day) * n_sec_per_day
    return start + datetime.timedelta(seconds=random_second)


def generate_temporal_point_within_boundaries() -> SpatioTemporalPoint:
    point = generate_point_in_area()
    while not check_point(point):
        point = generate_point_in_area()
    date = generate_random_date(start_date, end_date - duration_simulation)
    return SpatioTemporalPoint(lat=point.lat, lon=point.lon, date_time=date)


def check_point(point: SpatialPoint):
    """
    Check if the given point is in the specified area and not too close to a shore
    Args:
        point:

    Returns:

    """
    return lat_bottom <= point.lat.deg <= lat_top and lon_left <= point.lon.deg <= lon_right and True  # check_if_square_margin_to_shore()


# todo: to verify
def check_if_square_margin_to_shore(prob, point, margin_to_shore):
    # Step 1: get the bounding indices of the grid
    x_low_idx = (np.abs(prob.hindcast_data_source['grid_dict']['x_grid'] - (point[0] - margin_to_shore))).argmin()
    x_high_idx = (np.abs(prob.hindcast_data_source['grid_dict']['x_grid'] - (point[0] + margin_to_shore))).argmin()
    y_low_idx = (np.abs(prob.hindcast_data_source['grid_dict']['y_grid'] - (point[1] - margin_to_shore))).argmin()
    y_high_idx = (np.abs(prob.hindcast_data_source['grid_dict']['y_grid'] - (point[1] + margin_to_shore))).argmin()

    # Step 2: check if any of the land-mask contains land
    contains_land = np.any(
        prob.hindcast_data_source['grid_dict']['spatial_land_mask'][y_low_idx:y_high_idx, x_low_idx:x_high_idx])
    return not contains_land


def point_to_dict(point: Union[SpatialPoint, SpatioTemporalPoint], radius_in_m: int = 1) -> dict[str, any]:
    return {"lon_in_deg": point.lon.deg, "lat_in_deg": point.lat.deg} | \
           ({"datetime": point.date_time.strftime('%Y-%m-%dT%H:%M:%SZ')} if isinstance(point,
                                                                                       SpatioTemporalPoint) else {
               "radius_in_m": radius_in_m})


def problem_format(problem: tuple[SpatioTemporalPoint, SpatialPoint]):
    start, end = problem
    return {"initial_position": point_to_dict(start), "target": point_to_dict(end)}


def problems_to_yaml(problems: list[str], dir_name: str, filename: str):
    if dir_name is None:
        dir_name = "."
    if filename is None:
        filename = "problems"
    suffix = ".yaml"

    path = os.path.join(dir_name, filename + suffix)
    with open(path, 'w') as f:
        yaml.dump({"problems": [problem_format(problem) for problem in problems]}, f, sort_keys=False,
                  default_flow_style=False)
        return path


def main(dir_name=None, filename=None):
    print("VERIFY check_if_square_margin_to_shore")
    random.seed(seed)
    problems = []
    for _ in range(number_problems):
        start = generate_temporal_point_within_boundaries()
        end = generate_end_point(start.to_spatial_point())
        problems.append((start, end))

    path = problems_to_yaml(problems, dir_name=dir_name, filename=filename)
    print(f"problems exported to {path} = {os.path.abspath(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="file_path")
    parser.add_argument('--dir-name', dest='dir', type=str, help='path of the directory')
    parser.add_argument('--file-name', dest='file', type=str, help='filename of the list')
    args = parser.parse_args()
    main(args.dir, args.file)
