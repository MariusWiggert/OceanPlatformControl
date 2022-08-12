import string
import yaml
import os
import time
import datetime
from typing import Optional, List
from c3python import C3Python

from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint

class ArenaFactory:
    @staticmethod
    def create(
        scenario_name: string,
        problem: NavigationProblem = None,
        verbose: Optional[bool] = True
    ) -> Arena:
        if verbose:
            print(f'ArenaFactory: Creating Arena for {scenario_name}')

        with open(f'config/arena/{scenario_name}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Step 1: Connect to c3
        start = time.time()
        c3 = C3Python(
            url='https://dev01-seaweed-control.c3dti.ai',
            tenant='seaweed-control',
            tag='dev01',
            keyfile='setup/c3-rsa',
            username='jeanninj@berkeley.edu',
        ).get_c3()
        if verbose:
            print(f'ArenaFactory: Connect to c3 ({time.time() - start:.1f}s)')

        # Step 2: Download Hindcast
        if problem is not None and config['ocean_dict']['hindcast'] is not None and config['ocean_dict']['hindcast']['source']=='hindcast_files':
            start = time.time()
            ArenaFactory.download_required_files(
                c3=c3,
                archive_source=config['ocean_dict']['hindcast']['source_settings']['source'],
                archive_type=config['ocean_dict']['hindcast']['source_settings']['type'],
                download_folder=config['ocean_dict']['hindcast']['source_settings']['folder'],
                problem=problem
            )
            if verbose:
                print(f'ArenaFactory: Download Hindcast Files ({time.time() - start:.1f}s)')

        # Step 3: Download Forecast
        if problem is not None and config['ocean_dict']['forecast'] is not None and config['ocean_dict']['forecast']['source'] == 'forecast_files':
            start = time.time()
            ArenaFactory.download_required_files(
                c3=c3,
                archive_source=config['ocean_dict']['forecast']['source_settings']['source'],
                archive_type=config['ocean_dict']['forecast']['source_settings']['type'],
                download_folder=config['ocean_dict']['forecast']['source_settings']['folder'],
                problem=problem
            )
            if verbose:
                print(f'ArenaFactory: Download Forecast Files ({time.time() - start:.1f}s)')

        # Step 4: Create Arena
        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
            spatial_boundary=config['spatial_boundary'],
            verbose=verbose
        )

    @staticmethod
    def get_filelist(c3, archive_source, archive_type, time_interval: Optional[tuple[datetime.datetime, datetime.datetime]] = None):
        if time_interval is None:
            time_filter = ''
        else:
            start_min = f'{time_interval[0].replace(hour=0, minute=0, second=0, microsecond=0)}'
            start_max = f'{time_interval[1]}'
            time_filter = f' && subsetOptions.timeRange.start >= "{start_min}" && subsetOptions.timeRange.start <= "{start_max}"'

        if archive_source=='HYCOM' and archive_type=='forecast':
            archive_id = c3.HycomDataArchive.fetch(spec={'filter': 'description=="Full geo-spatial Hycom GOM: u,v at depth=4.0"'}).objs[0].fmrcArchive.id
            return c3.HycomFMRCFile.fetch(spec={'filter': f'archive=="{archive_id}" && status == "downloaded"{time_filter}', 'order': "ascending(subsetOptions.timeRange.start)"})
        elif archive_source=='HYCOM' and archive_type=='hindcast':
            archive_id = c3.HycomDataArchive.fetch(spec={'filter': 'description=="Full geo-spatial Hycom GOM: u,v at depth=4.0"'}).objs[0].hindcastArchive.id
            return c3.HycomHindcastFile.fetch(spec={'filter': f'archive=="{archive_id}" && status == "downloaded"{time_filter}', 'order': "ascending(subsetOptions.timeRange.start)"})
        elif archive_source=='Copernicus' and archive_type=='forecast':
            archive_id = c3.CopernicusDataArchive.fetch(spec={'filter': 'description=="Full GOM: utotal,vtotal at depth=0.49"'}).objs[0].fmrcArchive.id
            return c3.CopernicusFMRCFile.fetch(spec={'filter': f'archive=="{archive_id}" && status == "downloaded"{time_filter}', 'order': "ascending(subsetOptions.timeRange.start)"})
        elif archive_source=='Copernicus' and archive_type=='hindcast':
            archive_id = c3.CopernicusDataArchive.fetch(spec={'filter': 'description=="Full GOM: utotal,vtotal at depth=0.49"'}).objs[0].hindcastArchive.id
            return c3.CopernicusHindcastFile.fetch(spec={'filter': f'archive=="{archive_id}" && status == "downloaded"{time_filter}', 'order': "ascending(subsetOptions.timeRange.start)"})
        else:
            raise ValueError(f"Combination of archive_source={archive_source} and archive_type={archive_type} is not available.")

    @staticmethod
    def check_spacial_coverage(files, points: List[SpatialPoint]):
        """
            Helper function checking if points are in the spatial coverage of a file.
            Returns True if yes and False if not.
        """
        for file in files.objs:
            spacial_coverage = file.subsetOptions.geospatialCoverage
            for point in points:
                lon_covered = spacial_coverage.start.longitude < point.lon.deg < spacial_coverage.end.longitude
                lat_covered = spacial_coverage.start.latitude < point.lat.deg < spacial_coverage.end.latitude

                if lon_covered is False:
                    raise ValueError("The point {} is not covered by the longitude of the downloaded files.".format(point))
                if lat_covered is False:
                    raise ValueError("The point {} is not covered by the latitude of the downloaded files.".format(point))

    @staticmethod
    def download_filelist(c3, files, download_folder):
        # Step 1: Download Files thread-safe with atomic os.rename
        temp_folder = f'{download_folder}{os.getpid()}/'
        for file in files.objs:
            filename = os.path.basename(file.file.contentLocation)
            url = file.file.url
            filesize = file.file.contentLength
            if not os.path.exists(download_folder + filename) or os.path.getsize(download_folder + filename) != filesize:
                c3.Client.copyFilesToLocalClient(url, temp_folder)
                os.rename(temp_folder + filename, download_folder + filename)
                print(f'File downloaded: {filename}, {filesize}')
            else:
                os.system(f'touch {download_folder}{filename}')
            if os.path.getsize(download_folder + filename) != filesize:
                raise Exception(f"Downloaded forecast file with incorrect file size. Should be {filesize}B but is {os.path.getsize(download_folder + filename)}B.")
        os.system(f'rm -rf {temp_folder}')

        # Step 2: Only keep 100 most recent files to prevent full storage
        cached_files = [f'{download_folder}{file}' for file in os.listdir(download_folder)]
        cached_files = [file for file in cached_files if os.path.isfile(file)]
        cached_files.sort(key=os.path.getmtime, reverse=True)
        for file in cached_files[100:]:
            print('Deleting old forecast files:', file)
            os.remove(file)

    @staticmethod
    def download_required_files(c3, archive_source: str, archive_type: str, download_folder: str, problem: NavigationProblem):
        # Step 1: Find relevant files
        files = ArenaFactory.get_filelist(
            c3=c3,
            archive_source=archive_source,
            archive_type=archive_type,
            time_interval=(problem.start_state.date_time, problem.start_state.date_time + problem.timeout)
        )

        # Step 2: Check File Count
        if files.count < problem.timeout.days:
            raise Exception(f"Only {files.count} files in the database for t_0={problem.start_state.date_time} and problem.timeout.days={problem.timeout.days}.")

        # Step 3: Check Basic Spatial Coverage
        ArenaFactory.check_spacial_coverage(files, [problem.start_state.to_spatial_point(), problem.end_region])

        # Step 4: Download files thread-safe
        ArenaFactory.download_filelist(c3, files, download_folder)