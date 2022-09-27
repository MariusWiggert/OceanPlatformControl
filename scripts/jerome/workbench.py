# %%
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
import datetime

arena = ArenaFactory.create(scenario_name="gulf_of_mexico_Copernicus_forecast_and_hindcast")
# %%
dataset = arena.ocean_field.forecast_data_source.get_data_over_area(
    x_interval=[80, 81],
    y_interval=[80, 81],
    t_interval=[
        datetime.datetime(2021, 11, 20, 0, 10, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2021, 11, 20, 5, 10, 0, tzinfo=datetime.timezone.utc),
    ],
    spatial_resolution=1 / 10,
    temporal_resolution=3600,
)
print(dataset.to_array().to_numpy().shape)
print(dataset)
