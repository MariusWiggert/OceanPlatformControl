import bisect

import yaml
import math
import glob, os, imageio
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import netCDF4
import ocean_navigation_simulator.utils.plotting_utils as plot_utils
from ocean_navigation_simulator.utils.simulation_utils import get_current_data_subset_local_file, convert_to_lat_lon_time_bounds


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        x_0:
            The starting state, represented as (lon, lat) or (lat, lon, battery_level).
            Note that time is implemented as absolute time in POSIX.
        x_T:
            The target state, represented as (lon, lat).
            # TODO: currently we do point-2-point navigation though ultimately we'd like to do point to region
            this to be a set representation (point-2-region) because that is the more general formulation.
        t_0:
            A timezone aware datetime object of the absolute starting time of the platform at x_0

        platform_config_dict:
            A dict specifying the platform parameters, see the repos 'configs/platform.yaml' as example.
            If None: assumes python is run locally inside the OceanPlatformControl repo
                     where the data is stored under '/data' and the config under '/configs/platform.yaml'
                     => this is meant for local testing of functions only.

        # TO IMPLEMENT USAGE/FOR FUTURE
        forecast_delay_in_h:
            The hours of delay when a forecast becomes available
            e.g. forecast starts at 1st of Jan but only available from HYCOM 48h later on 3rd of January
        noise:
            # TODO: optionally implement a way to add noise to the hindcasts

        # Note sure about those guys yet...
        hindcast_file:
            Path to the nc4 files of forecasted ocean currents. Will be used as true currents (potentially adding noise)
        forecast_folder:
            Path to the nc4 files of forecasted ocean currents

        # TO REVIEW/THINK ABOUT
        x_t_tol:
            Radius around x_T that when reached counts as "target reached"
            # Note: not used currently as the sim config has that value too.
    """

    def __init__(self, x_0, x_T, t_0, hindcast_file, forecast_folder=None, forecast_delay_in_h=0.,
                 noise=None, x_t_tol=0.1, platform_config_dict=None):

        # load the respective fieldsets
        self.hindcast_file = hindcast_file  # because the simulator loads only subsetting of it

        time_len, time_range, x_range, y_range = self.extract_grid_dict_from_nc(hindcast_file)

        self.hindcast_grid_dict = {"gt_t_range": time_range,
                                   "gt_y_range": y_range,
                                   "gt_x_range": x_range,
                                   }
        # create forecast dict list with ranges & filename
        forecast_files_list = [forecast_folder + f for f in listdir(forecast_folder) if
                               (isfile(join(forecast_folder, f)) and f != '.DS_Store')]
        self.forecasts_dict = self.create_forecasts_dicts(forecast_files_list)

        # Log the problem specs
        print("GT fieldset from  {} to {}".format(datetime.utcfromtimestamp(
            self.hindcast_grid_dict['gt_t_range'][0]),
            datetime.utcfromtimestamp(self.hindcast_grid_dict['gt_t_range'][1])))
        print("GT Resolution of {} h".format(
            math.ceil((self.hindcast_grid_dict['gt_t_range'][1] - self.hindcast_grid_dict['gt_t_range'][0])
                      / (time_len * 3600))))
        print("Forecast files from {} to {}".format(datetime.utcfromtimestamp(
            self.forecasts_dict[0]['t_range'][0]),
            datetime.utcfromtimestamp(self.forecasts_dict[-1]['t_range'][0])))
        # get most recent forecast_idx for t_0
        for i, dic in enumerate(self.forecasts_dict):
            # Note: this assumes the dict is ordered according to time-values
            # which is true for now, but more complicated once we're in the C3 platform this should be done
            # in a better way
            if dic['t_range'][0] > t_0.timestamp() + forecast_delay_in_h * 3600:
                self.most_recent_forecast_idx = i - 1
                break

        if len(x_0) == 2:  # add 100% charge
            x_0 = x_0 + [1.]
        elif len(x_0) != 3:
            raise ValueError("x_0 should be (lat, lon, charge)")

        # add POSIX timestamp of t_0
        x_0 = x_0 + [t_0.timestamp()]
        self.x_0 = x_0
        self.x_T = x_T
        self.x_t_tol = x_t_tol
        self.forecast_delay_in_h = forecast_delay_in_h
        # self.most_recent_forecast_idx = self.check_current_files_provided()

        # Step 4: derive relative batter dynamics variables from config_dict
        if platform_config_dict is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # load in YAML
            with open(project_dir + '/configs/platform.yaml') as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
                platform_config_dict = config_data['platform_config']
        self.dyn_dict = self.derive_platform_dynamics(platform_config_dict)

    def __repr__(self):
        """Returns the string representation of a Problem, to be used for debugging.

        Returns:
            A String
        """
        return "Problem(x_0: {0}, x_T: {1})".format(self.x_0, self.x_T)

    def viz(self, time=None, video=False, filename=None, cut_out_in_deg=0.8, html_render=None):
        """Visualizes the Hindcast file with the ocean currents in a plot or a gif for a specific time or time range.

        Input Parameters:
        - time: the time to visualize the ocean currents as a datetime.datetime object if
                None, the visualization is at the t_0 time of the problem.
        - video: if True a matplotlib animation is created if filename is not None then it's saved, otherwise displayed
        - filename: a string for filepath and name with ending either '.gif' or '.mp4' under which it is saved
        - cut_out_in_deg: if None, the full fieldset is visualized, otherwise provide a float e.g. 0.5 to plot only
                a box of the x_0 and x_T including a 0.5 degrees outer buffer.

        Returns:
            None
        """

        t_interval, lat_bnds, lon_bnds = convert_to_lat_lon_time_bounds(self.x_0, self.x_T,
                                                                        deg_around_x0_xT_box=cut_out_in_deg,
                                                                        temp_horizon_in_h=None)
        # Step 0: get respective data subset from hindcast file
        grids_dict, u_data, v_data = get_current_data_subset_local_file(self.hindcast_file, t_interval, lat_bnds, lon_bnds)

        print("Note only the GT file is currently visualized")

        # define helper functions to add on top of current visualization

        def add_ax_func(ax, time=None, x_0=self.x_0[:2], x_T=self.x_T[:2]):
            del time
            ax.scatter(x_0[0], x_0[1], c='r', marker='o', s=200, label='start')
            ax.scatter(x_T[0], x_T[1], c='g', marker='*', s=200, label='goal')
            plt.legend(loc='upper right')

        # if we want to visualize with video
        if time is None and video:
            # create animation with extra func
            plot_utils.viz_current_animation(grids_dict['t_grid'], grids_dict, u_data, v_data,
                                             interval=200, ax_adding_func=add_ax_func, html_render=html_render,
                                             save_as_filename=filename)
        # otherwise plot static image
        else:
            if time is None:
                time = datetime.fromtimestamp(self.x_0[3])
            # plot underlying currents at time
            ax = plot_utils.visualize_currents(time.timestamp(), grids_dict, u_data, v_data, autoscale=True, plot=False)
            # add the start and goal position to the plot
            add_ax_func(ax)
            plt.show()

    def derive_platform_dynamics(self, platform_specs):
        """Derives the relative battery capacity dynamics (from 0-1) based on absolute physical values.
        Input:
            platform_specs      a dictionary containing the required platform specs.
        Returns:
            A dictionary of settings for the Problem, i.e. {'charge': __, 'energy': __, 'u_max': __}
        """

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
        if self.x_0[3] <= self.forecasts_dict[0]['t_range'][0] + self.forecast_delay_in_h * 3600.:
            raise ValueError("No forecast file available at the starting time t_0")
        # 2.1 check most recent file at t_0 for x_0
        times_available = [dict['t_range'][0] + self.forecast_delay_in_h * 3600. for dict in self.forecasts_dict]
        idx = bisect.bisect_right(times_available, self.x_0[3]) - 1
        if not in_interval(self.x_0[3], self.forecasts_dict[idx]['t_range']):
            raise ValueError("t_0 is not in the timespan of the most recent forecase file.")
        print("At starting time {}, most recent forecast available is from {} to {}.".format(
            datetime.utcfromtimestamp(self.x_0[3]),
            datetime.utcfromtimestamp(self.forecasts_dict[idx]['t_range'][0]),
            datetime.utcfromtimestamp(self.forecasts_dict[idx]['t_range'][1])))
        # Step 2: check location around x_0 for both hindcast & forecast (because interpolation doesn't check it..)                                                                                     ))
        if not in_interval(self.x_0[0], self.forecasts_dict[idx]['x_range']):
            raise ValueError("lon of x_0 is not in most recent forecast lon range.")
        if not in_interval(self.x_0[1], self.forecasts_dict[idx]['y_range']):
            raise ValueError("lat of x_0 is not in most recent forecast lon range.")
        return idx

    def extract_grid_dict_from_nc(self, file):
        """ Extracts the time, lat, and lon, dict from an nc_file."""
        f = netCDF4.Dataset(file)
        # get the time coverage in POSIX
        try:
            time_origin = datetime.strptime(f.variables['time'].__dict__['time_origin'] + ' +0000',
                                            '%Y-%m-%d %H:%M:%S %z')
        except:
            time_origin = datetime.strptime(f.variables['time'].__dict__['units'] + ' +0000',
                                            'hours since %Y-%m-%d %H:%M:%S.000 UTC %z')

        start_time_posix = (time_origin + timedelta(hours=f.variables['time'][0].data.tolist())).timestamp()
        end_time_posix = (time_origin + timedelta(hours=f.variables['time'][-1].data.tolist())).timestamp()
        # get the lat and lon intervals
        time_range = [start_time_posix, end_time_posix]
        y_range = [min(f.variables['lat'][:]), max(f.variables['lat'][:])]
        x_range = [min(f.variables['lon'][:]), max(f.variables['lon'][:])]
        time_vec_len = len(f.variables['time'][:].data)
        return time_vec_len, time_range, x_range, y_range

    def create_forecasts_dicts(self, forecast_files_list):
        """ Takes in a list of files and returns a list of tuples with:
        (start_time_posix, end_time_posix, grids, file) sorted according to start_time_posix
        """
        forecast_dicts = []
        for file in forecast_files_list:
            _, time_range, x_range, y_range = self.extract_grid_dict_from_nc(file)
            forecast_dicts.append({'t_range': time_range, 'x_range': x_range, 'y_range': y_range, 'file': file})
        # sort the tuples list
        forecast_dicts.sort(key=lambda dict: dict['t_range'][0])

        return forecast_dicts

# class WaypointTrackingProblem(Problem):
#     #TODO: not fit to new closed loop controller yet
#     """ Only difference is the added waypoints to the problem """
#
#     def __init__(self, real_fieldset, forecasted_fieldset, x_0, x_T, project_dir, waypoints,
#                  config_yaml='platform.yaml',
#                  fixed_time=None):
#         super().__init__(real_fieldset=real_fieldset,
#                          forecasted_fieldset=forecasted_fieldset,
#                          x_0=x_0,
#                          x_T=x_T,
#                          project_dir=project_dir,
#                          config_yaml=config_yaml,
#                          fixed_time=fixed_time)
#         self.waypoints = waypoints
#
#     @classmethod
#     def convert_problem(cls, problem, waypoints):
#         """ Given a problem, construct the corresponding WaypointTrackingProblem, with the same waypoints """
#         return WaypointTrackingProblem(real_fieldset=problem.real_fieldset,
#                                        forecasted_fieldset=problem.forecasted_fieldset,
#                                        x_0=problem.x_0,
#                                        x_T=problem.x_T,
#                                        project_dir=problem.project_dir,
#                                        waypoints=waypoints)
