import argparse
import datetime
import math
import os
import random
from typing import Union

import numpy as np
import xarray as xr
import yaml

from ocean_navigation_simulator.environment.PlatformState import SpatioTemporalPoint, SpatialPoint
from ocean_navigation_simulator.utils.units import Velocity, Distance

seed = 4  # 30031996
tile_radius = 1
delta_points = 1 / 12
lon_left, lon_right = -95.362841, -85.766062
lat_bottom, lat_top = 22.0, 27


def get_area_coordinates():
    return [(lon_left, lat_top), (lon_right, lat_top), (lon_left, lat_bottom),
            (lon_right, lat_bottom)]


# tl, tr, bl, br (lon, lat)
start_date = datetime.datetime(2022, 4, 1, 12, 00, 00)
end_date = datetime.datetime(2022, 4, 30, 12, 00, 00)
start_date_2 = datetime.datetime(2022, 6, 1, 12, 00, 00)  # June
end_date_2 = datetime.datetime(2022, 8, 31, 12, 00, 00)
all_intervals = [(start_date, end_date), (start_date_2, end_date_2)]

duration_simulation_default = datetime.timedelta(days=5)
max_velocity = Velocity(mps=1)
distance_start_end_point = max_velocity * duration_simulation_default
file_used_to_check_boundaries = "./data/april_may_data/forecast_Copernicus_april_may/cmems_mod_glo_phy_anfc_merged-uv_PT1H-i-2022-03-30T12:30:00Z-2022-03-30T12:30:00Z-2022-04-08T23:30:00Z.nc"


def generate_end_point(initial_point: SpatialPoint) -> SpatialPoint:
    angle = random.uniform(0, 2 * math.pi)
    new_point = SpatialPoint(Distance(deg=initial_point.lon.deg + math.cos(angle) * distance_start_end_point.deg),
                             Distance(deg=initial_point.lat.deg + math.sin(angle) * distance_start_end_point.deg))
    return new_point if check_point(new_point) else generate_end_point(initial_point)


def generate_point_in_area() -> SpatialPoint:
    p = None
    while p is None or not check_point(p):
        long = random.uniform(min(lon_left, lon_right), max(lon_left, lon_right))
        lat = random.uniform(min(lat_bottom, lat_top), max(lat_bottom, lat_top))
        p = SpatialPoint(Distance(deg=long), Distance(deg=lat))
    return p


def get_random_dates_among_list(dates, number_dates, duration_simulation):
    dates = np.array(dates)
    sec = np.array([(end - start).total_seconds() for start, end in dates])
    return [generate_random_date(*pair, duration_simulation) for pair in
            dates[np.random.choice(range(len(sec)), number_dates, p=sec / sec.sum())]]


def generate_random_date(start, end, duration_simulation):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start - duration_simulation
    n_sec_per_day = 24 * 60 * 60
    int_delta = (delta.days * n_sec_per_day) + delta.seconds
    random_second = random.randrange(int_delta)
    random_second = (random_second // n_sec_per_day) * n_sec_per_day
    return start + datetime.timedelta(seconds=random_second)


def generate_temporal_points_within_boundaries(numbers) -> SpatioTemporalPoint:
    points = [generate_point_in_area() for _ in range(numbers)]
    # date = generate_random_date(start_date, end_date - duration_simulation)
    dates = get_random_dates_among_list(all_intervals, numbers, duration_simulation_default)
    return [SpatioTemporalPoint(lat=point.lat, lon=point.lon, date_time=date) for point, date in zip(points, dates)]


def check_point(point: SpatialPoint):
    """
    Check if the given point is in the specified area and not too close to a shore
    Args:
        point:

    Returns:

    """
    return lat_bottom <= point.lat.deg <= lat_top and lon_left <= point.lon.deg <= lon_right and True  # check_if_square_margin_to_shore()


# Not used
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
    suffix = f"_{len(problems)}.yaml"

    path = os.path.join(dir_name, filename + suffix)
    with open(path, 'w') as f:
        yaml.dump({"problems": [problem_format(problem) for problem in problems]}, f, sort_keys=False,
                  default_flow_style=False)
        return path


def check_area_to_have_no_nan():
    global lon_left, lon_right, lat_top, lat_bottom

    lon_lat_xarray = xr.open_dataset(file_used_to_check_boundaries).isel(depth=0, time=0)
    len_lat = len(lon_lat_xarray["latitude"])
    print("start shrinking area")
    while np.isnan(lon_lat_xarray.sel(longitude=slice(lon_left - tile_radius, lon_right + tile_radius)).isel(
            latitude=len_lat // 2).to_array().to_numpy()).sum():
        lon_left += delta_points
        lon_right -= delta_points
    print("new lon boundaries: ", lon_left, lon_right)

    len_lon = len(lon_lat_xarray["longitude"])
    while np.isnan(lon_lat_xarray.sel(latitude=slice(lat_bottom - tile_radius, lat_top + tile_radius)).isel(
            longitude=len_lon // 2).to_array().to_numpy()).sum():
        lat_bottom += delta_points
        lat_top -= delta_points
    print("new lat boundaries: ", lat_bottom, lat_top)

    assert not np.isnan(lon_lat_xarray.sel(latitude=slice(lat_bottom - tile_radius, lat_top + tile_radius),
                                           longitude=slice(lon_left - tile_radius,
                                                           lon_right + tile_radius)).to_array().to_numpy()).sum()


def main(dir_name=None, filename=None, number_problems=40):
    print("VERIFY check_if_square_margin_to_shore")
    random.seed(seed)

    check_area_to_have_no_nan()

    # for _ in range(number_problems):
    starts = generate_temporal_points_within_boundaries(number_problems)
    problems = [(start, generate_end_point(start.to_spatial_point())) for start in starts]

    path = problems_to_yaml(problems, dir_name=dir_name, filename=filename)
    print(f"problems exported to {path} = {os.path.abspath(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="file_path")
    parser.add_argument('--dir-name', dest='dir', type=str, help='path of the directory')
    parser.add_argument('--file-name', dest='file', type=str, help='filename of the list')
    parser.add_argument('--num-problems', dest='problems', type=int, help='number of problems wanted')
    args = parser.parse_args()
    main(args.dir, args.file, args.problems)
