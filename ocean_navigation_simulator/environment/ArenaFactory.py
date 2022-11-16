import datetime
import logging
import os
import shutil
from typing import List, Optional
import contextlib

import mergedeep
import yaml

from ocean_navigation_simulator.data_sources.OceanCurrentSource.OceanCurrentSource import (
    get_grid_dict_from_file,
)
from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import get_c3, timing_logger

logger = logging.getLogger("arena.factory")


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
        scenario_file: str = None,
        scenario_name: str = None,
        scenario_config: Optional[dict] = {},
        problem: Optional[NavigationProblem] = None,
        points: Optional[List[SpatialPoint]] = None,
        x_interval: Optional[List[units.Distance]] = None,
        y_interval: Optional[List[units.Distance]] = None,
        t_interval: Optional[List[datetime.datetime]] = None,
        throw_exceptions: Optional[bool] = True,
        c3: Optional = None
    ) -> Arena:
        """
        Create an Arena with all needed configuration dictionaries.
        Main way
        If problem or t_interval is fed in and folder is specified, data is downloaded from C3 directly. Otherwise local files.
        Arguments:
            scenario_file: main way: specifying location of configuration.yaml
            scenario_name: (deprecated) specifying config/arena/scenario_name.yaml
            scenario_config: overwriting configuration from file

            problem: Optional[NavigationProblem] = None,
            points: Optional[List[SpatialPoint]] = None,
            x_interval: Optional[List[units.Distance]] = None,
            y_interval: Optional[List[units.Distance]] = None,
            t_interval: Optional[List[datetime.datetime]] = None,

            throw_exceptions: throw exceptions instead of printing errors
        Example:
            arena = ArenaFactory.create(
                scenario_file='config/arena/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml',
                problem=problem
            )
        """
        with timing_logger(
            f"ArenaFactory: Creating Arena for {scenario_name or scenario_file} ({{}})",
            logger,
            logging.INFO,
        ):
            # Step 1: Load Configuration from file
            if scenario_file is not None:
                with open(scenario_file) as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            elif scenario_name is not None:
                with open(f"config/arena/{scenario_name}.yaml") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                config = {}

            # Sep 2: Merge scenario_config, so we can overwrite the file settings
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
                        + datetime.timedelta(
                            seconds=config["casadi_cache_dict"]["time_around_x_t"] + config.get('timeout', 0)
                        ),
                    ]

            if x_interval is not None and y_interval is not None:
                points = [
                    SpatialPoint(lon=x_interval[0], lat=y_interval[0]),
                    SpatialPoint(lon=x_interval[1], lat=y_interval[1]),
                ]

            # Step 5: Download Hindcast
            if (t_interval is not None
                and config["ocean_dict"]["hindcast"] is not None
                and config["ocean_dict"]["hindcast"]["source"] == "hindcast_files"
            ):
                with timing_logger(
                    "Download Hindcast Files: {start} until {end} ({{}})".format(
                        start=t_interval[0].strftime("%Y-%m-%d %H-%M-%S"),
                        end=t_interval[-1].strftime("%Y-%m-%d %H-%M-%S"),
                    ),
                    logger,
                ):

                    if config["ocean_dict"]["hindcast"]["source_settings"].get("local", False):
                        files = ArenaFactory.find_hycom_files(
                            config["ocean_dict"]["hindcast"]["source_settings"]["folder"],
                            t_interval,
                        )
                    else:
                        files = ArenaFactory.download_required_files(
                            archive_source=config["ocean_dict"]["hindcast"]["source_settings"][
                                "source"
                            ],
                            archive_type=config["ocean_dict"]["hindcast"]["source_settings"][
                                "type"
                            ],
                            download_folder=config["ocean_dict"]["hindcast"]["source_settings"][
                                "folder"
                            ],
                            region=config["ocean_dict"]["hindcast"]["source_settings"].get(
                                "region", "GOM"
                            ),
                            t_interval=t_interval,
                            throw_exceptions=throw_exceptions,
                            points=points,
                            c3=c3
                        )

                    logger.debug(f"Hindcast Files: {files}")

                    # Modify source_settings to only consider selected files (prevent loading hundreds of files!)
                    config["ocean_dict"]["hindcast"]["source_settings"]["folder"] = files

            # Step 6: Download Forecast
            if (
                t_interval is not None
                and config["ocean_dict"]["forecast"] is not None
                and config["ocean_dict"]["forecast"]["source"] == "forecast_files"
            ):
                with timing_logger(
                    "Download Forecast Files: {start} until {end} ({{}})".format(
                        start=t_interval[0].strftime("%Y-%m-%d %H-%M-%S"),
                        end=t_interval[-1].strftime("%Y-%m-%d %H-%M-%S"),
                    ),
                    logger,
                ):
                    if config["ocean_dict"]["forecast"]["source_settings"].get("local", False):
                        files = ArenaFactory.find_copernicus_files(
                            config["ocean_dict"]["forecast"]["source_settings"]["folder"],
                            t_interval,
                        )
                    else:
                        files = ArenaFactory.download_required_files(
                            archive_source=config["ocean_dict"]["forecast"]["source_settings"][
                                "source"
                            ],
                            archive_type=config["ocean_dict"]["forecast"]["source_settings"][
                                "type"
                            ],
                            download_folder=config["ocean_dict"]["forecast"]["source_settings"][
                                "folder"
                            ],
                            region=config["ocean_dict"]["hindcast"]["source_settings"].get(
                                "region", "GOM"
                            ),
                            t_interval=t_interval,
                            throw_exceptions=throw_exceptions,
                            points=points,
                            c3=c3
                        )

                    logger.debug(f"Forecast Files: {files}")

                    # Modify source_settings to only consider selected files (prevent loading hundreds of files!)
                    config["ocean_dict"]["forecast"]["source_settings"]["folder"] = files

            # Step 7: Create Arena
            return Arena(
                casadi_cache_dict=config["casadi_cache_dict"],
                platform_dict=config["platform_dict"],
                ocean_dict=config["ocean_dict"],
                use_geographic_coordinate_system=config.get(
                    "use_geographic_coordinate_system", False
                ),
                solar_dict=config.get("solar_dict", None),
                seaweed_dict=config.get("seaweed_dict", None),
                spatial_boundary=config.get("spatial_boundary", None),
                timeout=config.get('timeout', None),
            )

    @staticmethod
    @contextlib.contextmanager
    def download_files(config, type, t_interval, points, c3=None, throw_exceptions=True):
        """Helper method to be run in C3 context manager."""
        try:
            if 'files' in config["ocean_dict"][type]['source']:
                ArenaFactory.download_required_files(
                    archive_source=config["ocean_dict"][type]["source_settings"][
                        "source"
                    ],
                    archive_type=config["ocean_dict"][type]["source_settings"]["type"],
                    download_folder=config["ocean_dict"][type]["source_settings"]["folder"],
                    t_interval=t_interval,
                    region=config["ocean_dict"]["area"],
                    throw_exceptions=throw_exceptions,
                    points=points,
                    c3=c3
                )
            yield True
        finally:
            if 'files' in config["ocean_dict"][type]['source']:
                try:
                    shutil.rmtree(config["ocean_dict"][type]["source_settings"]["folder"])
                except BaseException:
                    pass
            pass

    @staticmethod
    def find_hycom_files(path, t_interval):
        files = []

        start_min = t_interval[0] - datetime.timedelta(days=1)
        start_max = t_interval[1]

        for f in os.listdir(path):

            start = datetime.datetime.fromisoformat(f[28:47]).replace(tzinfo=datetime.timezone.utc)
            # end = datetime.datetime.fromisoformat(f[49:68]).replace(tzinfo=datetime.timezone.utc)

            if start_min <= start <= start_max:
                files.append(path + f)

        return files

    @staticmethod
    def find_copernicus_files(path, t_interval):
        files = []

        start_min = t_interval[0] - datetime.timedelta(days=1)
        start_max = t_interval[1]

        for f in os.listdir(path):
            start = datetime.datetime.fromisoformat(f[40:59]).replace(tzinfo=datetime.timezone.utc)
            # end = datetime.datetime.fromisoformat(f[61:80]).replace(tzinfo=datetime.timezone.utc)

            if start_min <= start <= start_max:
                files.append(path + f)

        return files

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
        c3: Optional = None,
    ) -> List[str]:
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
            c3: c3 object can be passed in directly, if not a c3 object is created
        Returns:
            list of downloaded files
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
            c3=c3
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
                logger.error(message)
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
                logger.error(message)

        # Step 3: Check Basic Spatial Coverage
        ArenaFactory._check_spacial_coverage(files, points)

        # Step 4: Download files thread-safe
        downloaded_files = ArenaFactory._download_filelist(
            files=files,
            download_folder=download_folder,
            throw_exceptions=throw_exceptions,
            c3=c3
        )

        return downloaded_files

    @staticmethod
    def get_filelist(
        archive_source: str,
        archive_type: str,
        region: Optional[str] = "GOM",
        t_interval: List[datetime.datetime] = None,
        c3: Optional = None
    ):
        """
        helper function to get a list of available files from c3
        Args:
            archive_source: one of [HYCOM, Copernicus]
            archive_type: one of [forecast, hindcast]
            region: one of [Region 1,  Region 2, Region 3, ... Region 7, GOM]. Exception: Region 1 is not unique for HYCOM
            t_interval: List of datetime with length 2. None allows to search in all available times.
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
            c3 = get_c3()
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
                "filter": f'contains(archive.{"description" if region=="GOM" else "name"}, "{region}") && status == "downloaded"{time_filter}',
                "order": "ascending(subsetOptions.timeRange.start)",
            }
        )

    @staticmethod
    def _download_filelist(files, download_folder, throw_exceptions, c3):
        """
        thread-safe download with corruption and file size check
        Arguments:
            files: c3.FecthResult object
            download_folder: folder to download files to
            throw_exceptions: if True throws exceptions for missing/corrupted files
        Returns:
            list of downloaded files
        """
        if c3 is None:
            c3 = get_c3()

        logger.info(f"Downloading {files.count} files to '{download_folder}'.")

        # Step 1: Sanitize Inputs
        if not download_folder.endswith("/"):
            download_folder += "/"
        os.makedirs(download_folder, exist_ok=True)

        # Step 2: Download Files thread-safe with atomic os.replace
        downloaded_files = []
        temp_folder = f"{download_folder}{os.getpid()}/"
        try:
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
                        # check valid xarray file and meta length
                        try:
                            grid_dict_list = get_grid_dict_from_file(
                                temp_folder + filename, currents="total"
                            )

                            if (
                                file.subsetOptions.timeRange.start < grid_dict_list["t_range"][0]
                                or grid_dict_list["t_range"][-1] < file.subsetOptions.timeRange.end
                            ):
                                error = "File shorter than declared in meta: filename={filename}, meta: [{ms},{me}], file: [{gs},{ge}]".format(
                                    filename=filename,
                                    ms=file.subsetOptions.timeRange.start.strftime(
                                        "%Y-%m-%d %H-%M-%S"
                                    ),
                                    me=file.subsetOptions.timeRange.end.strftime(
                                        "%Y-%m-%d %H-%M-%S"
                                    ),
                                    gs=grid_dict_list["t_range"][0].strftime("%Y-%m-%d %H-%M-%S"),
                                    ge=grid_dict_list["t_range"][-1].strftime("%Y-%m-%d %H-%M-%S"),
                                )

                        except Exception:
                            error = f"Corrupted file: '{filename}'."

                    if error and throw_exceptions:
                        raise CorruptedOceanFileException(error)
                    elif error:
                        os.remove(temp_folder + filename)
                        logger.warning(error)
                        continue

                    # Move thread-safe
                    os.replace(temp_folder + filename, download_folder + filename)
                    logger.info(f"File downloaded: '{filename}', {filesize/10e6:.1f}MB.")
                else:
                    # Path().touch()
                    os.system(f"touch {download_folder + filename}")
                    logger.info(f"File already downloaded: '{filename}', {filesize/10e6:.1f}MB.")

                downloaded_files.append(download_folder + filename)
        except BaseException:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise
        else:
            shutil.rmtree(temp_folder, ignore_errors=True)

        # Step 3: Only keep most recent files to prevent full storage
        KEEP = 1000  # ~ 100 * 100MB = 10GB
        cached_files = [f"{download_folder}{file}" for file in os.listdir(download_folder)]
        cached_files = [file for file in cached_files if os.path.isfile(file)]
        cached_files.sort(key=os.path.getmtime, reverse=True)
        for file in cached_files[KEEP:]:
            logger.info(f"Deleting old forecast file: '{file}'")
            os.remove(file)

        return downloaded_files

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
                        logger.error(
                            "The point {} is not covered by the longitude of the downloaded files. Available: lon [{},{}], lat[{},{}].".format(
                                point,
                                spacial_coverage.start.longitude,
                                spacial_coverage.end.longitude,
                                spacial_coverage.start.latitude,
                                spacial_coverage.end.latitude,
                            )
                        )

    @staticmethod
    def analyze_files(archive_source, archive_type, region, true_length=False):
        """
        analyze available files for specified source/type in a plot
        Arguments:
            archive_source: one of [HYCOM, Copernicus]
            archive_type: one of [forecast, hindcast]
            region: one of [Region 1,  Region 2, Region 3, ... Region 7, GOM]. Exception: Region 1 is not unique for HYCOM
            true_length: if yes *download all files* and check actual length of files using xarray
        """
        title = f"{archive_source.capitalize()} {archive_type.capitalize()} {region}"
        files = ArenaFactory.get_filelist(
            archive_source=archive_source, archive_type=archive_type, region=region
        )

        if files.count > 0:
            if true_length:
                download_folder = f"/home/ubuntu/{archive_source.lower()}_{archive_type.lower()}/"
                logger.info("Download to: " + download_folder)
                downloaded = ArenaFactory._download_filelist(
                    files, download_folder=download_folder, throw_exceptions=False
                )
                grid_dict_list = [get_grid_dict_from_file(f, currents="total") for f in downloaded]
                dates = [d["t_range"][0] for d in grid_dict_list]
                length = [
                    (d["t_range"][-1] - d["t_range"][0]).total_seconds() // 3600
                    for d in grid_dict_list
                ]
                deltas = [
                    (dates[i] - dates[i - 1]).total_seconds() / 3600 / 24
                    for i, d in enumerate(dates)
                    if i > 0
                ]
            else:
                dates = [file.subsetOptions.timeRange.start for file in files.objs]
                length = [
                    (
                        file.subsetOptions.timeRange.end - file.subsetOptions.timeRange.start
                    ).total_seconds()
                    // 3600
                    for file in files.objs
                ]
                deltas = [
                    (dates[i] - dates[i - 1]).total_seconds() / 3600 / 24
                    for i, d in enumerate(dates)
                    if i > 0
                ]
                for i, delta in enumerate(deltas):
                    if not delta == 1:
                        logger.warning(
                            "Missing Files after:",
                            os.path.basename(files.objs[i].file.contentLocation),
                            f"delta: {delta * 24}h",
                        )

            # Plot
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme()
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
            fig.suptitle(f"{title} ({files.count} Files)")

            axs[0, 0].plot(dates, length, marker=".")
            if true_length:
                axs[0, 0].set_title("Length of files in h (XArray)")
            else:
                axs[0, 0].set_title("Length of files in h (Meta)")
            axs[0, 0].tick_params(axis="x", labelrotation=45)
            axs[0, 1].hist(length)
            axs[0, 1].set_title("Histogram of length of files in h")

            axs[1, 0].plot(dates[:-1], deltas, marker=".")
            axs[1, 0].set_title("Days before next file.")
            axs[1, 0].tick_params(axis="x", labelrotation=45)
            axs[1, 1].hist(deltas)
            axs[1, 1].set_title("Histogram of days before next file")

            fig.tight_layout()
            plt.show()
        else:
            logger.warning(f"No files for {title}")
