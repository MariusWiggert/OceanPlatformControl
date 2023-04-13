import pytest
import xarray as xr
from numpy.testing import assert_approx_equal


@pytest.mark.parametrize(
    "filepath, lon, lat, expected_distance",
    [
        (
            "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_0.nc",
            -83,
            28.5,
            28.5,
        ),
        (
            "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_-150.nc",
            -90,
            25,
            227,
        ),
    ],
)
def test__bathymetry_distance_is_not_0(filepath, lon, lat, expected_distance):
    ds = xr.open_dataset(filepath)
    distance = ds["distance"].interp(lat=lat, lon=lon).data
    print(distance)
    assert_approx_equal(distance, expected_distance, significant=2)


@pytest.mark.parametrize(
    "filepath, lon, lat, expected_distance",
    [
        (
            "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_-150.nc",
            -83,
            28.5,
            0,
        ),
        (
            "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_0.nc",
            -81.16,
            25.22,
            0,
        ),
    ],
)
def test_bathymetry_distance_is_0(filepath, lon, lat, expected_distance):
    ds = xr.open_dataset(filepath)
    distance = ds["distance"].interp(lat=lat, lon=lon).data
    print(distance)
    assert distance == expected_distance
