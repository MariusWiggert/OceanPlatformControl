import yaml, os, netCDF4, abc
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt

# import from other utils
import ocean_navigation_simulator.utils.plotting_utils as plot_utils
from ocean_navigation_simulator.utils.simulation_utils import convert_to_lat_lon_time_bounds, get_current_data_subset, \
    get_grid_dict_from_file, spatio_temporal_interpolation


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        x_0:
            The starting state, represented as (lat, lon, battery_level).
            Note that time is implemented as absolute time in POSIX.
        x_T:
            The target state, represented as (lon, lat).
            # TODO: currently we do point-2-point navigation though ultimately we'd like to do point to region
            this to be a set representation (point-2-region) because that is the more general formulation.
        t_0:
            A timezone aware datetime object of the absolute starting time of the platform at x_0

        platform_config_dict:
            A dict specifying the platform parameters, see the repos 'configs/platform.yaml' as example.

        hindcasts_dicts and forecasts_dicts
            A list of dicts ordered according to the starting time-range.
            One dict for each hindcast/forecast file available. The dicts for each hindcast file contains:
            {'t_range': [<datetime object>, <datetime object>], 'file': <filepath>})

        plan_on_gt:
            if True we only use hindcast data and plan on hindcasts. If False we use forecast data for the planner.

        x_T_radius:
            Radius around x_T that when reached counts as "target reached"

        # TO IMPLEMENT USAGE/FOR FUTURE
        forecast_delay_in_h:
            The hours of delay when a forecast becomes available
            e.g. forecast starts at 1st of Jan but only available from HYCOM 48h later on 3rd of January
        noise:
            # TODO: optionally implement a way to add noise to the hindcasts
    """

    def __init__(self, x_0, x_T, t_0, platform_config_dict,
                 hindcast_folder, forecast_folder,
                 plan_on_gt=False, forecast_delay_in_h=0., x_T_radius= 0.1, noise=None):

        # Plan on GT
        self.plan_on_gt = plan_on_gt

        # Basic check of inputs
        if len(x_0) != 3 or len(x_T) != 2:
            raise ValueError("x_0 should be (lat, lon, battery) and x_T (lat, lon)")

        # check t_0
        if t_0.tzinfo is None:
            print("Assuming input t_0 is in UTC time.")
            t_0 = t_0.replace(tzinfo=timezone.utc)
        elif t_0.tzinfo != timezone.utc:
            raise ValueError("Please provide t_0 as UTC or naive datetime object.")

        # Log start, goal and forecast delay.
        self.t_0 = t_0
        self.x_0 = x_0 + [t_0.timestamp()]
        self.x_T = x_T
        self.x_T_radius = x_T_radius
        self.forecast_delay_in_h = forecast_delay_in_h

        # Initialize the data dicts with None
        self.hindcasts_dicts, self.hindcast_grid_dict, self.forecasts_dicts, self.most_recent_forecast_idx = [None] * 4
        # Initialize them using the folders provided
        self.update_data_dicts(hindcast_folder, forecast_folder)

        # Check data compatibility
        self.check_data_compatibility(t_0, [self.x_0[:2], self.x_T[:2]])

        # derive relative battery dynamics variables from config_dict
        # The if-clause is in case we want to specify it as path to a yaml
        if isinstance(platform_config_dict, str):
            # get the local project directory (assuming a specific folder structure)
            project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # load simulator sim_settings YAML
            with open(project_dir + '/configs/' + platform_config_dict) as f:
                platform_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.dyn_dict = self.derive_platform_dynamics(platform_config_dict)

        print(self)

    def __repr__(self):
        """Returns the string representation of a Problem, to be used for debugging.

        Returns:
            A String
        """
        Nav_string = "Navigate from {} at time {} to {}.".format(
            self.x_0[:3], datetime.fromtimestamp(self.x_0[3], timezone.utc), self.x_T)
        Sim_string = "Simulate with GT current files from {} to {}".format(
            self.hindcast_grid_dict['gt_t_range'][0],
            self.hindcast_grid_dict['gt_t_range'][1])
        if self.plan_on_gt:
            Plan_string = "Planning on GT."
        else:
            Plan_string = "Planning on {} Forecast files starting from {} to {}".format(
                len(self.forecasts_dicts),
                self.forecasts_dicts[0]['t_range'][0],
                self.forecasts_dicts[-1]['t_range'][0])
        return Nav_string + '\n' + Sim_string + '\n' + Plan_string

    def check_hindcast_comatibility(self, t, points):
        """Helper function to check if the Hindcast files cover all points at t_0.
        Input:
            t       datetime_object of time
            points  list of [lon, lat] points
        """

        # Step 1: check if t_0 is in the Hindcast time-range
        if not (self.hindcast_grid_dict['gt_t_range'][0] < t < self.hindcast_grid_dict['gt_t_range'][1]):
            raise ValueError("Hindcast files do not cover {}.".format(t))
        # Step 2: check if x_0 and x_T are in the spatial coverage of the Hindcast files
        for point in points:
            if not (self.hindcast_grid_dict['gt_x_range'][0] < point[0] < self.hindcast_grid_dict['gt_x_range'][1]):
                raise ValueError("Hindcast files does not contain {} in longitude range.".format(point))
            if not (self.hindcast_grid_dict['gt_y_range'][0] < point[1] < self.hindcast_grid_dict['gt_y_range'][1]):
                raise ValueError("Hindcast files does not contain {} in latitude range.".format(point))

    def check_forecast_comatibility(self, t, points):
        """Helper function to check if the most recent Forecast file covers all points at t_0.
        Input:
            t       datetime_object of time
            points  list of [lon, lat] points
        """

        # Step 1: check if at t_0 there's a forecast available.
        if not (self.forecasts_dicts[self.most_recent_forecast_idx]['t_range'][0] < t <
                self.forecasts_dicts[self.most_recent_forecast_idx]['t_range'][1]):
            raise ValueError("Most recent Forecast file does not cover {}.".format(t))
        # Step 2: check if x_0 and x_T are in the spatial coverage of the most recent Forecast file
        for point in points:
            if not (self.forecasts_dicts[self.most_recent_forecast_idx]['x_range'][0] < point[0] <
                    self.forecasts_dicts[self.most_recent_forecast_idx]['x_range'][1]):
                raise ValueError("Most recent Forecast file does not contain {} in longitude range.".format(point))
            if not (self.forecasts_dicts[self.most_recent_forecast_idx]['y_range'][0] < point[1] <
                    self.forecasts_dicts[self.most_recent_forecast_idx]['y_range'][1]):
                raise ValueError("Most recent Forecast file does not contain {} in latitude range.".format(point))

    def viz(self, time=None, video=False, filename=None, cut_out_in_deg=0.8,
            html_render=None, temp_horizon_viz_in_h=None, return_ax=False, plot_start_target=True):
        """Visualizes the Hindcast file with the ocean currents in a plot or a gif for a specific time or time range.

        Input Parameters:
        - time: the time to visualize the ocean currents as a datetime.datetime object if
                None, the visualization is at the t_0 time of the problem.
        - video: if True a matplotlib animation is created if filename is not None then it's saved, otherwise displayed
        - filename: a string for filepath and name with ending either '.gif' or '.mp4' under which it is saved
        - cut_out_in_deg: if None, the full fieldset is visualized, otherwise provide a float e.g. 0.5 to plot only
                a box of the x_0 and x_T including a 0.5 degrees outer buffer.
        - html_render: render settings for html, if None then html is displayed directly (for Jupyter)
                                if 'safari' it's opened in a safari page
        - temp_horizon_viz_in_h: if we render a video, for how long do we want the visualization to run in hours.

        Returns:
            None
        """

        # Step 0: Find the time, lat, lon bounds for data_subsetting
        t_interval, lat_interval, lon_interval = convert_to_lat_lon_time_bounds(self.x_0, self.x_T,
                                                                                deg_around_x0_xT_box=cut_out_in_deg,
                                                                                temp_horizon_in_h=temp_horizon_viz_in_h)

        print("Note only the GT hindcast data is currently visualized")
        # Step 1: get the data_subset for plotting
        grids_dict, u_data, v_data = get_current_data_subset(t_interval, lat_interval, lon_interval,
                                                             file_dicts=self.hindcasts_dicts,
                                                             max_temp_in_h=120)

        def add_ax_func(ax, time=None, x_0=self.x_0[:2], x_T=self.x_T[:2]):
            del time
            if plot_start_target:
                ax.scatter(x_0[0], x_0[1], c='r', marker='o', s=200, label='start')
                goal_circle = plt.Circle((x_T[0], x_T[1]), self.x_T_radius, color='g', fill=True, alpha=0.6, label='target')
                ax.add_patch(goal_circle)
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
                time = datetime.fromtimestamp(self.x_0[3], tz=timezone.utc)
            # plot underlying currents at time
            ax = plot_utils.visualize_currents(time.timestamp(), grids_dict, u_data, v_data,
                                               # figsize=figsize,
                                               autoscale=True, plot=False)
            # add the start and goal position to the plot
            add_ax_func(ax)
            if return_ax:
                return ax
            else:
                plt.show()

    def derive_hindcast_grid_dict(self):
        """Helper function to create the hindcast grid dict from a dict of multiple files self.hindcasts_dicts.
        Note: this currently assumes all files have the same spatial coverage, needs to be changed once
        we go for multiple regions.
        """

        # Basic sanity check
        if len(self.hindcasts_dicts) == 0:
            raise ValueError("No Hindcast files in hindcasts_dicts.")

        # get spatial coverage by reading in the first file
        f = netCDF4.Dataset(self.hindcasts_dicts[0]['file'])
        xgrid = f.variables['lon'][:].data
        ygrid = f.variables['lat'][:].data

        y_range = [ygrid[0], ygrid[-1]]
        x_range = [xgrid[0], xgrid[-1]]

        # get time_range
        # If one file it's simple
        if len(self.hindcasts_dicts) == 1:
            time_range = self.hindcasts_dicts[0]['t_range']
        else:  # multiple daily files
            # get time-range by iterating over the return elements that are consecutive/exactly 1h apart
            for idx in range(len(self.hindcasts_dicts) - 1):
                if self.hindcasts_dicts[idx]['t_range'][1] + timedelta(hours=1) \
                        == self.hindcasts_dicts[idx + 1]['t_range'][0]:
                    continue
                else:
                    break
            time_range = [self.hindcasts_dicts[0]['t_range'][0], self.hindcasts_dicts[idx + 1]['t_range'][1]]

        return {"gt_t_range": time_range, "gt_y_range": y_range, "gt_x_range": x_range}

    @staticmethod
    def get_file_dicts(folder):
        """ Creates an list of dicts ordered according to time available, one for each nc file available in folder.
        The dicts for each file contains:
        {'t_range': [<datetime object>, T], 'file': <filepath> ,'y_range': [min_lat, max_lat], 'x_range': [min_lon, max_lon]}
        """
        # get a list of files from the folder
        files_list = [folder + f for f in os.listdir(folder) if
                      (os.path.isfile(os.path.join(folder, f)) and f != '.DS_Store')]

        # iterate over all files to extract the grids and put them in an ordered list of dicts
        list_of_dicts = []
        for file in files_list:
            grid_dict = get_grid_dict_from_file(file)
            # append the file to it:
            grid_dict['file'] = file
            list_of_dicts.append(grid_dict)
        # sort the list
        list_of_dicts.sort(key=lambda dict: dict['t_range'][0])
        return list_of_dicts

    def update_data_dicts(self, hindcast_folder, forecast_folder):
        """Derive the file dicts again and new forecast_idx and do compatibility checks.
        Inputs:
            hindcast_folder/forecast_folder     local path to folder where all hindcast/forecast files are
            x_t                                 full state of simulator at current time (lon, lat, battery, POSIX time)
        """
        # Step 1: Update Hindcast Data
        self.hindcasts_dicts = self.get_file_dicts(hindcast_folder)
        self.hindcast_grid_dict = self.derive_hindcast_grid_dict()

        # Forecast Data
        if not self.plan_on_gt:
            self.forecasts_dicts = self.get_file_dicts(forecast_folder)
            self.most_recent_forecast_idx = self.get_most_recent_forecast_idx()

    def check_data_compatibility(self, t, points):
        """Check if given forecast and hindcasts contain all points at time t."""
        self.check_hindcast_comatibility(t, points)
        if not self.plan_on_gt:
            self.check_forecast_comatibility(t, points)

    def get_most_recent_forecast_idx(self):
        """Get the index of the most recent forecast available t_0."""
        # Filter on the list to get all files where t_0 is contained.
        dics_containing_t_0 = list(
            filter(lambda dic: dic['t_range'][0] < self.t_0 < dic['t_range'][1], self.forecasts_dicts))
        # Basic Sanity Check if this list is empty no file contains t_0
        if len(dics_containing_t_0) == 0:
            raise ValueError("None of the forecast files contains t_0.")
        # As the dict is time-ordered we simple need to find the idx of the last one in the dics_containing_t_0
        for idx, dic in enumerate(self.forecasts_dicts):
            if dic['t_range'][0] == dics_containing_t_0[-1]['t_range'][0]:
                return idx

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
        charge_factor =  (platform_specs['solar_panel_size'] * platform_specs['solar_efficiency']) / cap_in_joule
        platform_dict = {
            # multiplied by Irridance (lat, lon, time) this is the relative battery charge per second
            'solar_factor': charge_factor,
            # multiplied by u^3 this is the relative battery discharge per second
            'energy': energy_coeff,
            'u_max': platform_specs['u_max']
        }

        return platform_dict
