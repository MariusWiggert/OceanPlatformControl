import datetime
import os
import shutil
from pathlib import Path
from typing import List, Optional

import mergedeep
import xarray as xr
import yaml

from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import get_c3, timing


# TODO: change to use loggers


class OceanFileException(Exception):
    def __repr__(self):
        return str(self)


class MissingOceanFileException(OceanFileException):
    pass


class CoverageOceanFileException(OceanFileException):
    pass


class CorruptedOceanFileException(OceanFileException):
    pass


class ArenaFactory:
    """Factory to create an arena with specific settings and download the needed files from C3 storage."""

    @staticmethod
    def create(
            scenario_name: str = None,
            scenario_file: str = None,
            folder_scenario: str = "config/arena/",
            scenario_config: Optional[dict] = {},
            problem: Optional[NavigationProblem] = None,
            points: Optional[List[SpatialPoint]] = None,
            x_interval: Optional[List[units.Distance]] = None,
            y_interval: Optional[List[units.Distance]] = None,
            t_interval: Optional[List[datetime.datetime]] = None,
            verbose: Optional[int] = 10,
            c3: Optional = None,
    ) -> Arena:
        """
        If problem or t_interval is fed in, data is downloaded from C3 directly. Otherwise local files.
        """
        with timing(
                f"ArenaFactory: Creating Arena for {scenario_name or scenario_file} ({{:.1f}}s)",
                verbose,
        ):
            # Step 1: Load Configuration from file
            if scenario_file is not None:
                with open(scenario_file) as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            elif scenario_name is not None:
                with open(os.path.join(folder_scenario, f"{scenario_name}.yaml")) as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                config = {}

            # Sep 2: Merge scenario_config so we can overwrite the file settings
            # https://mergedeep.readthedocs.io/en/latest/
            mergedeep.merge(config, scenario_config)

            # Step 3: Add Spatial Boundary
            if x_interval is not None and y_interval is not None:
                config["spatial_boundary"] = {
                    "x": [x_interval[0], x_interval[1]],
                    "y": [y_interval[0], y_interval[1]],
                }

            # Step 4: Add point to check for coverage
            if problem is not None:
                points = [problem.start_state.to_spatial_point(), problem.end_region]
                if t_interval is None:
                    t_interval = [
                        problem.start_state.date_time,
                        problem.start_state.date_time
                        + problem.timeout
                        + datetime.timedelta(
                            seconds=config["casadi_cache_dict"]["time_around_x_t"]
                        ),
                    ]
            elif x_interval is not None and y_interval is not None:
                points = [
                    SpatialPoint(lon=x_interval[0], lat=y_interval[0]),
                    SpatialPoint(lon=x_interval[1], lat=y_interval[1]),
                ]

            # Step 5: Download Hindcast
            if (
                    t_interval is not None
                    and config["ocean_dict"]["hindcast"] is not None
                    and config["ocean_dict"]["hindcast"]["source"] == "hindcast_files"
            ):
                with timing("ArenaFactory: Download Hindcast Files ({:.1f}s)", verbose):
                    ArenaFactory.download_files(
                        config=config,
                        type="hindcast",
                        t_interval=t_interval,
                        points=points,
                        verbose=verbose,
                        c3=c3,
                    )

            # Step 6: Download Forecast
            if (
                    t_interval is not None
                    and config["ocean_dict"]["forecast"] is not None
                    and config["ocean_dict"]["forecast"]["source"] == "forecast_files"
            ):
                with timing("ArenaFactory: Download Forecast Files ({:.1f}s)", verbose):
                    ArenaFactory.download_files(
                        config=config,
                        type="forecast",
                        t_interval=t_interval,
                        points=points,
                        verbose=verbose,
                        c3=c3,
                    )

            # Step 7: Create Arena
            return Arena(
                casadi_cache_dict=config.get("casadi_cache_dict", {}),
                platform_dict=config["platform_dict"],
                ocean_dict=config["ocean_dict"],
                use_geographic_coordinate_system=config["use_geographic_coordinate_system"],
                solar_dict=config["solar_dict"],
                seaweed_dict=config["seaweed_dict"],
                spatial_boundary=config.get("spatial_boundary", {}),
            )

    @staticmethod
    def download_files(config, type, t_interval, points, verbose, c3=None):
        """Helper method to be run in C3 context manager."""
        ArenaFactory.download_required_files(
            archive_source=config["ocean_dict"][type]["source_settings"]["source"],
            archive_type=config["ocean_dict"][type]["source_settings"]["type"],
            download_folder=config["ocean_dict"][type]["source_settings"]["folder"],
            t_interval=t_interval,
            region=config["ocean_dict"]["area"],
            throw_exceptions=True,
            points=points,
            verbose=verbose,
            c3=c3,
        )

    # TODO: automatically select best region depending on given points
    @staticmethod
    def download_required_files(
            archive_source: str,
            archive_type: str,
            download_folder: str,
            t_interval: List[datetime.datetime],
            region: str = "GOM",
            points: Optional[List[SpatialPoint]] = None,
            throw_exceptions: bool = False,
            verbose: Optional[int] = 10,
            c3: Optional = None,
    ) -> int:
        """
        helper function for thread-safe download of available current files from c3
        Args:
            archive_source: one of [HYCOM, Copernicus]
            archive_type: one of [forecast, hindcast]
            download_folder: path on disk to download the files e.g. /tmp/hycom_hindcast/
            t_interval: List of datetime with length 2.
            region: one of [Region 1,  Region 2, Region 3, ... Region 7, GOM]. Exception: Region 1 is not unique for HYCOM
            points: List of SpatialPoints to check for file coverage
            throw_exceptions: throw exceptions for missing or corrupted files
            verbose: silence print statements with 0
            c3: c3 object can be passed in directly, if not a c3 object is created
        Returns:
            amount of files found
        Examples:
            ArenaFactory.download_required_files(
                archive_source='hycom',
                archive_type='forecast',
                download_folder='/tmp/copernicus_forecast/',
                t_interval=[datetime(year=2022, month=4, day=1), datetime(year=2022, month=4, day=10)]
            )
        """
        # Step 1: Find relevant files
        files = ArenaFactory.get_filelist(
            archive_source=archive_source,
            archive_type=archive_type,
            region=region,
            t_interval=t_interval,
            verbose=verbose,
            c3=c3,
        )

        # Step 2: Check File Count
        if files.count == 0:
            message = "No files in the database for {archive_source}, {archive_type}, {region} and t_0={t_0} and t_T={t_T} ".format(
                archive_source=archive_source,
                archive_type=archive_type,
                region=region,
                t_0=t_interval[0],
                t_T=t_interval[1],
            )
            if throw_exceptions:
                raise MissingOceanFileException(message)
            else:
                print(message)
                return 0
        elif files.count < (t_interval[1] - t_interval[0]).days + 1:
            message = "Only {count}/{expected} files in the database for {archive_source}, {archive_type}, {region} and t_0={t_0} and t_T={t_T}: {filenames}".format(
                count=files.count,
                expected=(t_interval[1] - t_interval[0]).days + 1,
                archive_source=archive_source,
                archive_type=archive_type,
                region=region,
                t_0=t_interval[0],
                t_T=t_interval[1],
                filenames="".join(
                    [f"\n- {os.path.basename(f.file.contentLocation)}" for f in files.objs]
                ),
            )
            if throw_exceptions:
                raise MissingOceanFileException(message)
            else:
                print(message)

        # Step 3: Check Basic Spatial Coverage
        ArenaFactory._check_spacial_coverage(files, points)

        # Step 4: Download files thread-safe
        ArenaFactory._download_filelist(
            files=files,
            download_folder=download_folder,
            throw_exceptions=throw_exceptions,
            verbose=verbose,
            c3=c3,
        )

        return files.count

    @staticmethod
    def get_filelist(
            archive_source: str,
            archive_type: str,
            region: Optional[str] = "GOM",
            t_interval: List[datetime.datetime] = None,
            verbose: Optional[int] = 10,
            c3: Optional = None,
    ):
        """
        helper function to get a list of available files from c3
        Args:
            archive_source: one of [HYCOM, Copernicus]
            archive_type: one of [forecast, hindcast]
            region: one of [Region 1,  Region 2, Region 3, ... Region 7, GOM]. Exception: Region 1 is not unique for HYCOM
            t_interval: List of datetime with length 2. None allows to search in all available times.
            verbose: silence print statements with 0
        Return:
            c3.FetchResult where objs contains the actual files
        Examples:
            files = ArenaFactory.get_filelist('copernicus', 'forecast', 'Region 1')
            files = ArenaFactory.get_filelist('copernicus', 'forecast', 'Region 2')
            files = ArenaFactory.get_filelist('copernicus', 'forecast', 'GOM')
            files = ArenaFactory.get_filelist('hycom', 'forecast', 'Region 2')
            files = ArenaFactory.get_filelist('hycom', 'hindcast', 'Region 3')
        """
        # Step 1: Sanitize Inputs
        if archive_source.lower() not in ["hycom", "copernicus"]:
            raise ValueError(
                f"archive_source {archive_source} invalid choose from: hycom, copernicus."
            )
        if archive_type.lower() not in ["forecast", "hindcast"]:
            raise ValueError(
                f"archive_type {archive_type} invalid choose from: forecast, hindcast."
            )
        if region not in [
            "GOM",
            "Region 1",
            "Region 2",
            "Region 3",
            "Region 4",
            "Region 5",
            "Region 6",
            "Region 7",
        ]:
            raise ValueError(f"Region {region} invalid.")

        # Step 2: Find c3 type
        if c3 is None:
            c3 = get_c3(verbose - 1)
        archive_types = {"forecast": "FMRC", "hindcast": "Hindcast"}
        c3_file_type = getattr(
            c3, f"{archive_source.capitalize()}{archive_types[archive_type.lower()]}File"
        )

        # Step 3: Execute Query
        if t_interval is not None:
            # substracting 1 day is more safe in case the file doesn't start at midnight (e.g. Copernicus)
            start_min = f"{t_interval[0] - datetime.timedelta(days=1)}"
            start_max = f"{t_interval[1]}"
            time_filter = f' && subsetOptions.timeRange.start >= "{start_min}" && subsetOptions.timeRange.start <= "{start_max}"'
        else:
            # accepting t_interval = None allows to download the whole file list for analysis
            time_filter = ""

        return c3_file_type.fetch(
            spec={
                "filter": f'contains(archive.description, "{region}") && status == "downloaded"{time_filter}',
                "order": "ascending(subsetOptions.timeRange.start)",
            }
        )

    @staticmethod
    def _download_filelist(
            files, download_folder, throw_exceptions, verbose: Optional[int] = 10, c3: Optional = None
    ):
        """thread-safe download with corruption and file size check"""
        if c3 is None:
            c3 = get_c3(verbose - 1)

        # Step 1: Sanitize Inputs
        if not download_folder.endswith("/"):
            download_folder += "/"
        os.makedirs(download_folder, exist_ok=True)

        # Step 2: Download Files thread-safe with atomic os.replace
        try:
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

                    error = False
                    # check file size match
                    if os.path.getsize(temp_folder + filename) != filesize:
                        error = "Incorrect file size ({filename}). Should be {filesize}B but is {actual_filesize}B.".format(
                            filename=filename,
                            filesize=filesize,
                            actual_filesize=os.path.getsize(download_folder + filename),
                        )
                    else:
                        # check valid xarray file
                        try:
                            xr.open_mfdataset(temp_folder + filename)
                        except Exception:
                            error = f"Corrupted file: {filename})."

                    if error and throw_exceptions:
                        shutil.rmtree(temp_folder, ignore_errors=True)
                        raise CorruptedOceanFileException(error)
                    elif error:
                        os.remove(temp_folder + filename)
                        print(error)
                        continue

                    # Move thread-safe
                    os.replace(temp_folder + filename, download_folder + filename)
                    if verbose > 0:
                        print(f"File downloaded: {filename}, {filesize}.")
                else:
                    Path(download_folder + filename).touch()
                    if verbose > 0:
                        print(f"File already downloaded: {filename}, {filesize}.")
        except BaseException:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise
        else:
            shutil.rmtree(temp_folder, ignore_errors=True)

        # Step 3: Only keep most recent files to prevent full storage
        KEEP = 100  # ~ 100 * 100MB = 10GB
        cached_files = [f"{download_folder}{file}" for file in os.listdir(download_folder)]
        cached_files = [file for file in cached_files if os.path.isfile(file)]
        cached_files.sort(key=os.path.getmtime, reverse=True)
        for file in cached_files[KEEP:]:
            if verbose > 0:
                print("Deleting old forecast file:", file)
            os.remove(file)

    @staticmethod
    def _check_spacial_coverage(files, points: List[SpatialPoint]):
        """
        Helper function checking if all points are in the spatial coverage of a file.
        Returns True if yes and False if not.
        """
        if points is not None:
            for file in files.objs:
                spacial_coverage = file.subsetOptions.geospatialCoverage
                for point in points:
                    lon_covered = (
                            spacial_coverage.start.longitude
                            <= point.lon.deg
                            <= spacial_coverage.end.longitude
                    )
                    lat_covered = (
                            spacial_coverage.start.latitude
                            <= point.lat.deg
                            <= spacial_coverage.end.latitude
                    )

                    if not lon_covered or not lat_covered:
                        raise CoverageOceanFileException(
                            "The point {} is not covered by the longitude of the downloaded files. Available: lon [{},{}], lat[{},{}].".format(
                                point,
                                spacial_coverage.start.longitude,
                                spacial_coverage.end.longitude,
                                spacial_coverage.start.latitude,
                                spacial_coverage.end.latitude,
                            )
                        )
