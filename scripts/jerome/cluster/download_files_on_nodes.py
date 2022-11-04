import datetime
import os

import ray

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.utils import cluster_utils

t_interval = [
    # Copernicus FC: 2022-04 until today, HYCOM Hindcast: 2021-09 until today
    datetime.datetime(year=2022, month=4, day=8, tzinfo=datetime.timezone.utc),
    datetime.datetime(year=2022, month=10, day=9, tzinfo=datetime.timezone.utc),
]


@ray.remote(resources={"Node": 1})
def download_all_files(i):
    "purge download temp folders"
    os.system("rm -rf /home/ubuntu/copernicus_forecast/*/; rm -rf /home/ubuntu/hycom_hindcast/*/")
    ArenaFactory.download_required_files(
        archive_source="hycom",
        archive_type="hindcast",
        download_folder="/home/ubuntu/hycom_hindcast",
        t_interval=t_interval,
        verbose=0,
    )
    ArenaFactory.download_required_files(
        archive_source="copernicus",
        archive_type="forecast",
        download_folder="/home/ubuntu/copernicus_forecast",
        t_interval=t_interval,
        verbose=0,
    )
    print(f"Finished Node {i}")


nodes, _, _ = cluster_utils.init_ray()


ray.get([download_all_files.remote(i) for i in range(len(nodes))])

print("Finished")
