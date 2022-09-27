import yaml
import os
import datetime
from typing import Optional, List

from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import cluster_utils
from ocean_navigation_simulator.utils import units


class ArenaFactory:
    """Factory to create an arena with specific settings and download the needed files from C3 storage."""

    @staticmethod
    def create(
        scenario_name: str,
        problem: Optional[NavigationProblem] = None,
        points: Optional[List[SpatialPoint]] = None,
        x_interval: Optional[List[units.Distance]] = None,
        y_interval: Optional[List[units.Distance]] = None,
        t_interval: Optional[List[datetime.datetime]] = None,
        verbose: Optional[int] = 0,
    ) -> Arena:
        """If problem or t_interval is fed in, data is downloaded from C3 directly. Otherwise local files."""
        with units.timing(f"ArenaFactory: Creating Arena for {scenario_name} ({{:.1f}}s)", verbose):
            # Step 1: Load Configuration
            with open(f"config/arena/{scenario_name}.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            if x_interval is not None and y_interval is not None:
                config["spatial_boundary"] = {
                    "x": [x_interval[0], x_interval[1]],
                    "y": [y_interval[0], y_interval[1]],
                }
            if problem is not None:
                points = [problem.start_state.to_spatial_point(), problem.end_region]
                t_interval = [
                    problem.start_state.date_time,
                    problem.start_state.date_time + problem.timeout,
                ]
            elif x_interval is not None and y_interval is not None:
                points = [
                    SpatialPoint(lon=x_interval[0], lat=y_interval[0]),
                    SpatialPoint(lon=x_interval[1], lat=y_interval[1]),
                ]

            # Step 2: Download Hindcast
            if (
                t_interval is not None
                and config["ocean_dict"]["hindcast"] is not None
                and config["ocean_dict"]["hindcast"]["source"] == "hindcast_files"
            ):
                with units.timing("ArenaFactory: - Download Hindcast Files ({:.1f}s)", verbose):
                    ArenaFactory.download_required_files(
                        archive_source=config["ocean_dict"]["hindcast"]["source_settings"][
                            "source"
                        ],
                        archive_type=config["ocean_dict"]["hindcast"]["source_settings"]["type"],
                        download_folder=config["ocean_dict"]["hindcast"]["source_settings"][
                            "folder"
                        ],
                        t_interval=t_interval,
                        points=points,
                        verbose=verbose,
                    )

            # Step 3: Download Forecast
            if (
                t_interval is not None
                and config["ocean_dict"]["forecast"] is not None
                and config["ocean_dict"]["forecast"]["source"] == "forecast_files"
            ):
                with units.timing("ArenaFactory: - Download Forecast Files ({:.1f}s)", verbose):
                    ArenaFactory.download_required_files(
                        archive_source=config["ocean_dict"]["forecast"]["source_settings"][
                            "source"
                        ],
                        archive_type=config["ocean_dict"]["forecast"]["source_settings"]["type"],
                        download_folder=config["ocean_dict"]["forecast"]["source_settings"][
                            "folder"
                        ],
                        t_interval=t_interval,
                        points=points,
                        verbose=verbose,
                    )

            # Step 4: Create Arena
            return Arena(
                casadi_cache_dict=config["casadi_cache_dict"],
                platform_dict=config["platform_dict"],
                ocean_dict=config["ocean_dict"],
                use_geographic_coordinate_system=config["use_geographic_coordinate_system"],
                solar_dict=config["solar_dict"],
                seaweed_dict=config["seaweed_dict"],
                spatial_boundary=config["spatial_boundary"],
            )

    @staticmethod
    def get_filelist(c3, archive_source, archive_type, t_interval: List[datetime.datetime]):
        start_min = f"{t_interval[0].replace(hour=0, minute=0, second=0, microsecond=0)}"
        start_max = f"{t_interval[1]}"
        time_filter = f' && subsetOptions.timeRange.start >= "{start_min}" && subsetOptions.timeRange.start <= "{start_max}"'

        if archive_source == "HYCOM" and archive_type == "forecast":
            archive_id = (
                c3.HycomDataArchive.fetch(
                    spec={"filter": 'description=="Full geo-spatial Hycom GOM: u,v at depth=4.0"'}
                )
                .objs[0]
                .fmrcArchive.id
            )
            return c3.HycomFMRCFile.fetch(
                spec={
                    "filter": f'archive=="{archive_id}" && status == "downloaded"{time_filter}',
                    "order": "ascending(subsetOptions.timeRange.start)",
                }
            )
        elif archive_source == "HYCOM" and archive_type == "hindcast":
            archive_id = (
                c3.HycomDataArchive.fetch(
                    spec={"filter": 'description=="Full geo-spatial Hycom GOM: u,v at depth=4.0"'}
                )
                .objs[0]
                .hindcastArchive.id
            )
            return c3.HycomHindcastFile.fetch(
                spec={
                    "filter": f'archive=="{archive_id}" && status == "downloaded"{time_filter}',
                    "order": "ascending(subsetOptions.timeRange.start)",
                }
            )
        elif archive_source == "Copernicus" and archive_type == "forecast":
            archive_id = (
                c3.CopernicusDataArchive.fetch(
                    spec={"filter": 'description=="Full GOM: utotal,vtotal at depth=0.49"'}
                )
                .objs[0]
                .fmrcArchive.id
            )
            return c3.CopernicusFMRCFile.fetch(
                spec={
                    "filter": f'archive=="{archive_id}" && status == "downloaded"{time_filter}',
                    "order": "ascending(subsetOptions.timeRange.start)",
                }
            )
        elif archive_source == "Copernicus" and archive_type == "hindcast":
            archive_id = (
                c3.CopernicusDataArchive.fetch(
                    spec={"filter": 'description=="Full GOM: utotal,vtotal at depth=0.49"'}
                )
                .objs[0]
                .hindcastArchive.id
            )
            return c3.CopernicusHindcastFile.fetch(
                spec={
                    "filter": f'archive=="{archive_id}" && status == "downloaded"{time_filter}',
                    "order": "ascending(subsetOptions.timeRange.start)",
                }
            )
        else:
            raise ValueError(
                f"Combination of archive_source={archive_source} and archive_type={archive_type} is not available."
            )

    @staticmethod
    def check_spacial_coverage(files, points: List[SpatialPoint]):
        """
        Helper function checking if points are in the spatial coverage of a file.
        Returns True if yes and False if not.
        """
        if points is not None:
            for file in files.objs:
                spacial_coverage = file.subsetOptions.geospatialCoverage
                for point in points:
                    lon_covered = (
                        spacial_coverage.start.longitude
                        < point.lon.deg
                        < spacial_coverage.end.longitude
                    )
                    lat_covered = (
                        spacial_coverage.start.latitude
                        < point.lat.deg
                        < spacial_coverage.end.latitude
                    )

                    if lon_covered is False:
                        raise ValueError(
                            "The point {} is not covered by the longitude of the downloaded files.".format(
                                point
                            )
                        )
                    if lat_covered is False:
                        raise ValueError(
                            "The point {} is not covered by the latitude of the downloaded files.".format(
                                point
                            )
                        )

    @staticmethod
    def download_filelist(c3, files, download_folder, verbose: Optional[int] = 0):
        """Helper Function to download files from a list in a thread safe manner."""
        # Step 1: Download Files thread-safe with atomic os.rename
        temp_folder = f"{download_folder}{os.getpid()}/"
        for file in files.objs:
            filename = os.path.basename(file.file.contentLocation)
            url = file.file.url
            filesize = file.file.contentLength
            if (
                not os.path.exists(download_folder + filename)
                or os.path.getsize(download_folder + filename) != filesize
            ):
                c3.Client.copyFilesToLocalClient(url, temp_folder)
                os.replace(temp_folder + filename, download_folder + filename)
                if verbose > 0:
                    print(f"File downloaded: {filename}, {filesize}")
            else:
                os.system(f"sudo touch {download_folder}{filename}")
            if os.path.getsize(download_folder + filename) != filesize:
                raise Exception(
                    f"Downloaded forecast file with incorrect file size. Should be {filesize}B but is {os.path.getsize(download_folder + filename)}B."
                )
        os.system(f"rm -rf {temp_folder}")

        # Step 2: Only keep 100 most recent files to prevent full storage
        cached_files = [f"{download_folder}{file}" for file in os.listdir(download_folder)]
        cached_files = [file for file in cached_files if os.path.isfile(file)]
        cached_files.sort(key=os.path.getmtime, reverse=True)
        for file in cached_files[100:]:
            if verbose > 0:
                print("Deleting old forecast files:", file)
            os.remove(file)

    @staticmethod
    def download_required_files(
        archive_source: str,
        archive_type: str,
        download_folder: str,
        t_interval: Optional[List[datetime.datetime]] = [],
        points: Optional[List[SpatialPoint]] = None,
        verbose: Optional[int] = 0,
    ):
        # Step 1: Find relevant files
        files = ArenaFactory.get_filelist(
            c3=cluster_utils.get_c3(verbose - 1),
            archive_source=archive_source,
            archive_type=archive_type,
            t_interval=t_interval,
        )

        # Step 2: Check File Count
        if files.count < (t_interval[1] - t_interval[0]).days + 1:
            raise Exception(
                f"Only {files.count} files in the database for t_0={t_interval[0]} and t_T={t_interval[1]}."
            )

        # Step 3: Check Basic Spatial Coverage
        ArenaFactory.check_spacial_coverage(files, points)

        # Step 4: Download files thread-safe
        ArenaFactory.download_filelist(
            cluster_utils.get_c3(verbose - 1), files, download_folder, verbose
        )
