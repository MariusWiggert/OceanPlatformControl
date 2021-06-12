import bisect

import yaml
import parcels as p
import math
import numpy as np
import glob, os, imageio
from src.utils import hycom_utils
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
from dateutil import tz
import netCDF4


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        x_0:
            The starting state, represented as (lon, lat) or (lat, lon, battery_level).
            Note that time is implemented as absolute time in POSIX.
        x_T:
            The target state, represented as (lon, lat).
            # TODO: currently we do point-2-point navigation though ultimately we'd like
            this to be a set representation (point-2-region) because that is the more general formulation.
        t_0:
            A timezone aware datetime object of the absolute starting time of the platform at x_0
        # TODO: implement flexible loading of the nc4 files outside of here
        hindcast_file:
            Path to the nc4 files of forecasted ocean currents. Will be used as true currents (potentially adding noise)
        forecast_folder:
            Path to the nc4 files of forecasted ocean currents
        forecast_delay_in_h:
            The hours of delay when a forecast becomes available
            e.g. forecast starts at 1st of Jan but only available from HYCOM 48h later on 3rd of January
        noise:
            # TODO: optionally implement a way to add noise to the hindcasts
        x_t_tol:
            Radius around x_T that when reached counts as "target reached"
            # Note: not used currently as the sim config has that value too.
        config_yaml:
            A YAML file for the platform configurations.
        fixed_time:
            datetime object of the fixed time for the hindcast_file as GT
        project dir:
            Only needed if the data is stored outside the repo
    """

    def __init__(self, x_0, x_T, t_0, hindcast_file, forecast_folder=None, forecast_delay_in_h=0.,
                 noise=None, x_t_tol=0.1, config_yaml='platform.yaml',
                 fixed_time=None, project_dir=None):

        if project_dir is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # load the respective fieldsets
        self.hindcast_file = hindcast_file  # because the simulator loads only subsetting of it
        self.hindcast_fieldset = hycom_utils.get_hycom_fieldset(hindcast_file)  # this loads in all for plotting

        # get the gt_field_grid
        hindcast_posix_times = self.hindcast_fieldset.gridset.grids[0].timeslices[0].astype(datetime) / (10 ** 9)

        self.hindcast_grid_dict = {"gt_t_range": [hindcast_posix_times[0], hindcast_posix_times[-1]],
                                   "gt_y_range": [min(self.hindcast_fieldset.gridset.grids[0].lat),
                                                  max(self.hindcast_fieldset.gridset.grids[0].lat)],
                                   "gt_x_range": [min(self.hindcast_fieldset.gridset.grids[0].lon),
                                                  max(self.hindcast_fieldset.gridset.grids[0].lon)],
                                   }
        # create forecast dict list with ranges & filename
        forecast_files_list = [forecast_folder + f for f in listdir(forecast_folder) if isfile(join(forecast_folder, f))]
        self.forecasts_dict = self.create_forecasts_dicts(forecast_files_list)

        if fixed_time is not None:
            self.fixed_time = fixed_time
            print("Fixed-time fieldset currently not implemented")
            raise NotImplementedError()
        else:  # variable time
            self.fixed_time = None
            print("GT fieldset from  {} to {}".format(self.hindcast_fieldset.time_origin,
                                                      self.hindcast_fieldset.gridset.grids[0].timeslices[0][-1]))
            print("GT Resolution of {} h".format(math.ceil(self.hindcast_fieldset.gridset.grids[0].time[1] / 3600)))
            print("Forecast files from {} to {}".format(datetime.utcfromtimestamp(
                self.forecasts_dict[0]['t_range'][0]), datetime.utcfromtimestamp(self.forecasts_dict[-1]['t_range'][0])))

        if len(x_0) == 2:  # add 100% charge
            x_0 = x_0 + [1.]
        elif len(x_0) != 3:
            raise ValueError("x_0 should be (lat, lon, charge)")

        # add POSIX timestamp of t_0
        x_0 = x_0 + [t_0.timestamp()]
        self.x_0 = x_0
        self.x_T = x_T
        self.project_dir = project_dir
        self.forecast_delay_in_h = forecast_delay_in_h
        self.most_recent_forecast_idx = self.check_current_files_provided()
        self.dyn_dict = self.derive_platform_dynamics(config=config_yaml)

    def __repr__(self):
        """Returns the string representation of a Problem, to be used for debugging.

        Returns:
            A String
        """
        return "Problem(x_0: {0}, x_T: {1})".format(self.x_0, self.x_T)

    def viz(self, time=None, gif=False):
        """Visualizes the problem in a plot or a gif.

        Returns:
            None
        """
        print("Note only the GT file is currently visualized")
        pset = p.ParticleSet.from_list(fieldset=self.hindcast_fieldset,  # the fields on which the particles are advected
                                       pclass=p.ScipyParticle,  # the type of particles (JITParticle or ScipyParticle)
                                       lon=[self.x_0[0], self.x_T[0]],  # a vector of release longitudes
                                       lat=[self.x_0[1], self.x_T[1]],  # a vector of release latitudes
                                       )

        if self.fixed_time is None and time is None and gif:
            # Step 1: create the images
            pset = p.ParticleSet.from_list(fieldset=self.hindcast_fieldset,  # the fields on which the particles are advected
                                           pclass=p.ScipyParticle,
                                           # the type of particles (JITParticle or ScipyParticle)
                                           lon=[self.x_0[0], self.x_T[0]],  # a vector of release longitudes
                                           lat=[self.x_0[1], self.x_T[1]],  # a vector of release latitudes
                                           )
            for i, time in enumerate(self.hindcast_fieldset.gridset.grids[0].time_full):
                # under the assumption that x is a Position
                pset.show(savefile=self.project_dir + '/viz/pics_2_gif/particles' + str(i).zfill(2),
                          field='vector', land=True,
                          vmax=1.0, show_time=time)

            # Step 2: compile to gif
            file_list = glob.glob(self.project_dir + "/viz/pics_2_gif/*")
            file_list.sort()

            gif_file = self.project_dir + '/viz/gifs/' + "var_prob_viz" + '.gif'
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in file_list:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)
            print("saved gif as " + "var_prob_viz")
        elif self.fixed_time is None and time is None and not gif:
            pset.show(field='vector', show_time=0)
        elif time is None:  # fixed time index
            pset.show(field='vector', show_time=self.fixed_time)
        else:
            pset.show(field='vector', show_time=time)

    def derive_platform_dynamics(self, config):
        """Derives battery capacity dynamics for the specific dt.

        Args:
            config:
                See config_yaml in class docstring

        Returns:
            A dictionary of settings for the Problem, i.e. {'charge': __, 'energy': __, 'u_max': __}
        """

        # load in YAML
        with open(self.project_dir + '/config/' + config) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            platform_specs = config_data['platform_config']

        # derive calculation
        cap_in_joule = platform_specs['battery_cap'] * 3600
        energy_coeff = (platform_specs['drag_factor'] * (1 / platform_specs['motor_efficiency'])) / cap_in_joule
        charge_factor = platform_specs['avg_solar_power'] / cap_in_joule
        platform_dict = {'charge': charge_factor, 'energy': energy_coeff, 'u_max': platform_specs['u_max']}

        return platform_dict

    def check_current_files_provided(self):
        def in_interval(x, interval):
            if interval[0] <= x <= interval[1]:
                return True
            else:
                return False
        # Step 1: check gt_file
        if not in_interval(self.x_0[3], self.hindcast_grid_dict['gt_t_range']):
            raise ValueError("t_0 is not in hindcast fieldset time range.")
        if not in_interval(self.x_0[0], self.hindcast_grid_dict['gt_x_range']):
            raise ValueError("lon of x_0 is not in hindcast fieldset lon range.")
        if not in_interval(self.x_0[1], self.hindcast_grid_dict['gt_y_range']):
            raise ValueError("lat of x_0 is not in hindcast fieldset lon range.")
        if not in_interval(self.x_T[0], self.hindcast_grid_dict['gt_x_range']):
            raise ValueError("lon of x_T is not in hindcast fieldset lon range.")
        if not in_interval(self.x_T[1], self.hindcast_grid_dict['gt_y_range']):
            raise ValueError("lat of x_T is not in hindcast fieldset lon range.")

        # Step 2: check forecast file
        # 2.1 check if at start_time any forecast is available
        if self.x_0[3] <= self.forecasts_dict[0]['t_range'][0] + self.forecast_delay_in_h *3600.:
            raise ValueError("No forecast file available at the starting time t_0")
        # 2.1 check most recent file at t_0 for x_0
        times_available = [dict['t_range'][0] + self.forecast_delay_in_h *3600. for dict in self.forecasts_dict]
        idx = bisect.bisect_right(times_available, self.x_0[3]) - 1
        if not in_interval(self.x_0[3], self.forecasts_dict[idx]['t_range']):
            raise ValueError("t_0 is not in the timespan of the most recent forecase file.")
        print("At starting time {}, most recent forecast available is from {} to {}.".format(datetime.utcfromtimestamp(self.x_0[3]),
                                                                                             datetime.utcfromtimestamp(self.forecasts_dict[idx]['t_range'][0]),
                                                                                             datetime.utcfromtimestamp(self.forecasts_dict[idx]['t_range'][1])))
        # Step 2: check location around x_0 for both hindcast & forecast (because interpolation doesn't check it..)                                                                                     ))
        if not in_interval(self.x_0[0], self.forecasts_dict[idx]['x_range']):
            raise ValueError("lon of x_0 is not in most recent forecast lon range.")
        if not in_interval(self.x_0[1], self.forecasts_dict[idx]['y_range']):
            raise ValueError("lat of x_0 is not in most recent forecast lon range.")
        return idx

    def create_forecasts_dicts(self, forecast_files_list):
        """ Takes in a list of files and returns a list of tuples with:
        (start_time_posix, end_time_posix, grids, file) sorted according to start_time_posix
        """
        forecast_dicts = []
        for file in forecast_files_list:
            f = netCDF4.Dataset(file)
            # get the time coverage in POSIX
            time_origin = datetime.strptime(f.variables['time'].__dict__['units'] + ' +0000',
                                            'hours since %Y-%m-%d %H:%M:%S.000 UTC %z')
            start_time_posix = (time_origin + timedelta(hours=f.variables['time'][0].data.tolist())).timestamp()
            end_time_posix = (time_origin + timedelta(hours=f.variables['time'][-1].data.tolist())).timestamp()
            # get the lat and lon intervals
            time_range = [start_time_posix, end_time_posix]
            y_range = [min(f.variables['lat'][:]), max(f.variables['lat'][:])]
            x_range = [min(f.variables['lon'][:]), max(f.variables['lon'][:])]
            forecast_dicts.append({'t_range': time_range, 'x_range': x_range, 'y_range': y_range, 'file':file})
        # sort the tuples list
        forecast_dicts.sort(key=lambda dict: dict['t_range'][0])

        return forecast_dicts



class WaypointTrackingProblem(Problem):
    #TODO: not fit to new closed loop controller yet
    """ Only difference is the added waypoints to the problem """

    def __init__(self, real_fieldset, forecasted_fieldset, x_0, x_T, project_dir, waypoints,
                 config_yaml='platform.yaml',
                 fixed_time=None):
        super().__init__(real_fieldset=real_fieldset,
                         forecasted_fieldset=forecasted_fieldset,
                         x_0=x_0,
                         x_T=x_T,
                         project_dir=project_dir,
                         config_yaml=config_yaml,
                         fixed_time=fixed_time)
        self.waypoints = waypoints

    @classmethod
    def convert_problem(cls, problem, waypoints):
        """ Given a problem, construct the corresponding WaypointTrackingProblem, with the same waypoints """
        return WaypointTrackingProblem(real_fieldset=problem.real_fieldset,
                                       forecasted_fieldset=problem.forecasted_fieldset,
                                       x_0=problem.x_0,
                                       x_T=problem.x_T,
                                       project_dir=problem.project_dir,
                                       waypoints=waypoints)
