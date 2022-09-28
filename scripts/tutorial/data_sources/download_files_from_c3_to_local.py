import datetime
import os

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

# TODO: somehow doesn't work, ask jerome who coded this function!

# Download data for specific time interval
t_0 = datetime.datetime(2022, 4, 4, 23, 30, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=2)]
os.makedirs("data/downloaded_hindcast_files", exist_ok=True)

ArenaFactory.download_required_files(
    archive_source="HYCOM",
    archive_type="hindcast",
    download_folder="tmp/downloaded_hindcast_files",
    t_interval=t_interval,
    verbose=2,
)
