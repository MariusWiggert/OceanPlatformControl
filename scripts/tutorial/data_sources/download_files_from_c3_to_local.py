import datetime
import os

from ocean_navigation_simulator.data_sources.C3Downloader import C3Downloader
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

# There are multiple ways of downloading files from c3.
# They use the same functions under the hood but slightly different interfaces, the top one is more general!
#%% Using the C3 Downloader Object
folder = "data/downloaded_hindcast_files"
c3_downloader = C3Downloader()
time_interval = [
    datetime.datetime(2022, 4, 21, 12, 0, 0),
    datetime.datetime(2022, 4, 22, 12, 0, 0),
]
files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", time_interval)
c3_downloader.download_files(files, folder)

#%% Using the Arena Factory helper function
# Download data for specific time interval. THIS ONLY WORKS FOR GULF OF MEXICO (needs modification for other areas)
folder = "data/downloaded_hindcast_files/"
t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=2)]
os.makedirs(folder, exist_ok=True)

ArenaFactory.download_required_files(
    archive_source="HYCOM",
    archive_type="hindcast",
    download_folder=folder,
    t_interval=t_interval,
    verbose=2,
)
