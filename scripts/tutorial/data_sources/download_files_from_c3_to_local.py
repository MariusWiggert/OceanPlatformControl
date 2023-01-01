# %% imports
import datetime

from ocean_navigation_simulator.data_sources.C3Downloader import C3Downloader
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

time_interval = [
    datetime.datetime(2022, 11, 28, 00, 0, 0),
    datetime.datetime(2022, 12, 20, 23, 59, 59),
]

# There are two ways of downloading files from c3.
# They use the same functions under the hood but slightly different interfaces and checks.
#%% Using the C3 Downloader Object
# Option 1: just download all available files (a bit lighter)
# folder = "data/HYCOM/Hindcast/"
# c3_downloader = C3Downloader()
# files = c3_downloader.get_files_list("HYCOM", "hindcast", "Region 3", time_interval)
# c3_downloader.download_files(files, folder)

#%% Using the Arena Factory helper function
# Option 2: Download filed and do a bunch of checking in terms of spatial coverage, corrupt files, and thread safe.
# Download data for specific time interval.
folder = "path_name"

ArenaFactory.download_required_files(
    archive_source="Copernicus",
    archive_type="forecast",
    region="Region 3",
    download_folder=folder,
    t_interval=time_interval,
    keep_newest_days=1000,
)
