import datetime

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

# ArenaFactory.download_required_files(
#     archive_source="copernicus",
#     archive_type="forecast",
#     region="GOM",
#     download_folder='/tmp/copernicus_forecast/',
#     t_interval=[datetime.datetime(2022, 1, 1), datetime.datetime(2022, 10, 30)],
# )

ArenaFactory.download_required_files(
    archive_source="hycom",
    archive_type="hindcast",
    region="GOM",
    download_folder="/tmp/corrupted_files",
    t_interval=[datetime.datetime(2021, 10, 1), datetime.datetime(2021, 10, 10)],
)
