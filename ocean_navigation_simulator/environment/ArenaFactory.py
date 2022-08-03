import os
import string
import time
from typing import Optional

import yaml
from c3python import C3Python

from ocean_navigation_simulator.environment.Arena import Arena
from ocean_navigation_simulator.environment.Problem import Problem


class ArenaFactory:
    @staticmethod
    def create(scenario_name: string = None, file: string = None, pid=None, timing: Optional[bool] = False) -> Arena:
        if scenario_name:
            with open(f'scenarios/{scenario_name}.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        elif file:
            with open(file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError('Specify scenario name or file.')

        if pid is not None and config['ocean_dict']['forecast'] is not None and config['ocean_dict']['forecast']['source'] == 'forecast_files':
            config['ocean_dict']['forecast']['source_settings']['folder'] += f'{pid}/'

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
    def download_files(problem: Problem, n_days_ahead=6,  timing: Optional[bool] = False):
        # Step 1: Connect to c3
        c3 = C3Python(
            url='https://dev01-seaweed-control.c3dti.ai',
            tenant='seaweed-control',
            tag='dev01',
            keyfile='./setup/c3-rsa',
            username='jeanninj@berkeley.edu',
        ).get_c3()

        # Step 2: Download Files
        pid = os.getpid()
        os.system(f'rm -rf /tmp/hycom_forecast/{pid}')
        c3.HycomDataArchive.DownloadForecastFilesToLocal(
            HycomDataArchive=c3.HycomDataArchive.fetch(spec={'filter': 'dataset=="GOMu0.04/expt_90.1m000"'}).objs[0],
            t_0=problem.start_state.date_time,
            x_0=[problem.start_state.lon.deg, problem.start_state.lat.deg],
            x_T=[problem.end_region.lon.deg, problem.end_region.lat.deg],
            n_days_ahead=n_days_ahead,
            local_folder=f'/tmp/hycom_forecast/{pid}'
        )