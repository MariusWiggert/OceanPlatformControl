import string
import yaml
import os
import datetime
from typing import Optional
from c3python import C3Python

from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint

class ArenaFactory:
    @staticmethod
    def create(scenario_name: string = None, file: string = None, timing: Optional[bool] = False) -> Arena:
        if scenario_name:
            with open(f'config/scenarios/{scenario_name}.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        elif file:
            with open(file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError('Specify scenario name or file.')

        return Arena(
            sim_cache_dict=config['sim_cache_dict'],
            platform_dict=config['platform_dict'],
            ocean_dict=config['ocean_dict'],
            use_geographic_coordinate_system=config['use_geographic_coordinate_system'],
            solar_dict=config['solar_dict'],
            seaweed_dict=config['seaweed_dict'],
            spatial_boundary=config['spatial_boundary'],
            timing=timing
        )

    @staticmethod
    def download_hycom_forecast(problem: Problem, n_days_ahead: int = 6, download_folder: Optional[str] = '/tmp/hycom_forecast'):
        # Step 1: get c3 type
        c3 = C3Python(
            url='https://dev01-seaweed-control.c3dti.ai',
            tenant='seaweed-control',
            tag='dev01',
            keyfile='setup/c3-rsa',
            username='jeanninj@berkeley.edu',
        ).get_c3()

        # Step 2: Find relevant files
        archive_id = c3.HycomDataArchive.fetch(spec = { 'filter': 'dataset=="GOMu0.04/expt_90.1m000"' }).objs[0].fmrcArchive.id
        start_min = f'{problem.start_state.date_time - datetime.timedelta(days=1)}'
        start_max = f'{(problem.start_state.date_time + datetime.timedelta(days=n_days_ahead)).replace(hour=23, minute=59, second=0, microsecond=0)}'
        files = c3.HycomFMRCFile.fetch(spec={
            'filter': f'archive=="{archive_id}" && status == "downloaded" && subsetOptions.timeRange.start >= "{start_min}" && subsetOptions.timeRange.start <= "{start_max}"',
            'order': "ascending(subsetOptions.timeRange.start)",
        })

        # Step 3: Check File Count
        if files.count < n_days_ahead:
            raise Exception(f"Only {files.count} forecast files in the database for t_0={problem.start_state.date_time} and n_days_ahead={n_days_ahead}.")
        # Step 4: Check Basic Spatial Coverage
        for file in files.objs:
            ArenaFactory.check_spacial_coverage(file.subsetOptions.geospatialCoverage, problem.start_state.to_spatial_point())
            ArenaFactory.check_spacial_coverage(file.subsetOptions.geospatialCoverage, problem.end_region)

        # Step 5: Download Files thread-safe
        temp_folder = f'{download_folder}/{os.getpid()}'
        for file in files.objs:
            filename = os.path.basename(file.file.contentLocation)
            url = file.file.url
            filesize = file.file.contentLength
            if not os.path.exists(download_folder + '/' + filename) or os.path.getsize(download_folder + '/' + filename) != filesize:
                c3.Client.copyFilesToLocalClient(url, temp_folder)
                os.rename(temp_folder + '/' + filename, download_folder + '/' + filename)
            else:
                os.system(f'sudo touch {download_folder}/{filename}')
            if os.path.getsize(download_folder + '/' + filename) != filesize:
                raise Exception(f"Downloaded forecast file with incorrect file size. Should be {filesize}B but is {os.path.getsize(download_folder + '/' + filename)}B.")
        os.system(f'rm -rf {temp_folder}')

        # Step 6: Only keep 50 most recent files to prevent full storage
        cached_files = [f'{download_folder}/{file}' for file in os.listdir(download_folder)]
        cached_files = [file for file in cached_files if os.path.isfile(file)]
        cached_files.sort(key=os.path.getmtime, reverse=True)
        for file in cached_files[50:]:
            print('Deleting:', file)
            os.remove(file)

    @staticmethod
    def check_spacial_coverage(spacial_coverage, point: SpatialPoint):
        """
            Helper function checking if the point is in the spatial coverage of a file.
            Returns True if yes and False if not.
        """
        lon_covered = spacial_coverage.start.longitude < point.lon.deg < spacial_coverage.end.longitude
        lat_covered = spacial_coverage.start.latitude < point.lat.deg < spacial_coverage.end.latitude

        if lon_covered is False:
            raise ValueError("The point {} is not covered by the longitude of the downloaded file.".format(point))
        if lat_covered is False:
            raise ValueError("The point {} is not covered by the latitude of the downloaded file.".format(point))