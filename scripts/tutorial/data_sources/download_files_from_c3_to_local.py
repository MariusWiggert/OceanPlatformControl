import datetime
import logging

from ocean_navigation_simulator.data_sources.C3Downloader import C3Downloader
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.utils.misc import set_arena_loggers

time_interval = [
    datetime.datetime(2022, 4, 21, 12, 0, 0),
    datetime.datetime(2022, 4, 22, 12, 0, 0),
]

# Get Info Message from all arena components
set_arena_loggers(logging.INFO)

# There are two ways of downloading files from c3.
# They use the same functions under the hood but slightly different interfaces and checks.
#%% Using the C3 Downloader Object
# Option 1: just download all available files (a bit lighter)
folder = "data/downloaded_hindcast_files"
c3_downloader = C3Downloader()
files = c3_downloader.get_files_list("Copernicus", "forecast", "Region 1", time_interval)
c3_downloader.download_files(files, folder)

#%% Using the Arena Factory helper function
# Option 2: Download filed and do a bunch of checking in terms of spatial coverage, corrupt files, and thread safe.
# Download data for specific time interval.
folder = "data/downloaded_hindcast_files/"

ArenaFactory.download_required_files(
    archive_source="Copernicus",
    archive_type="forecast",
    region="Region 1",
    download_folder=folder,
    t_interval=time_interval,
)
