import yaml, os
# import h5netcdf.legacyapi as netCDF4
# import netCDF4
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ocean_navigation_simulator import utils


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        x_0:
            The starting state, represented as (lat, lon, battery_level).
        t_0:
            A timezone aware datetime object of the absolute starting time of the platform at x_0
        x_T:
            The spatial center of the target state, represented as (lon, lat), together with x_T_radius it's a 2D region.
        x_T_radius:
            Radius around x_T that when reached counts as "target reached"
        platform_config_dict:
            A dict specifying the platform parameters, see the repos 'configs/platform.yaml' as example.
        hindcast_source and forecast_source:
            A dict object, always containing "data_source_type" and "data_source" which gives the information
            where to get the data for the forecast (what the planner sees) and the hindcast (what the simulator uses).
            Currently, we have three "data_source_type"'s implemented:
            {"single_nc_file", "multiple_daily_nc_files", "cop_opendap", "analytical_function"}
            They all require different inputs for "content"
            "single_nc_file", "multiple_daily_nc_files": expect a path to the folder of the file(s)
            "cop_opendap": expects a dict with the specifications to establish the connection.
                       For now only Copernicus is implemented (but HYCOM can be in the future too).
                       {'USERNAME': 'mmariuswiggert', 'PASSWORD': 'tamku3-qetroR-guwneq',
                       'DATASET_ID': 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh'}
            "analytical_function": A reference to a function with the signature u, v = analytical_function(x,y,t)
                                   This can be packaged with functools.partial to have different random seeds.
        plan_on_gt:
            if True we only use hindcast data and plan on hindcasts. If False we use forecast data for the planner.

        # TO IMPLEMENT USAGE/FOR FUTURE
        forecast_delay_in_h:
            The hours of delay when a forecast becomes available
            e.g. forecast starts at 1st of Jan but only available from HYCOM 48h later on 3rd of January
        noise:
            # TODO: optionally implement a way to add noise to the hindcasts
    """

    def __init__(self, x_0, x_T, t_0, platform_config_dict,
                 hindcast_source, forecast_source,
                 plan_on_gt=False, x_T_radius=0.1):

        # Basic check of inputs
        if len(x_0) != 3 or len(x_T) != 2:
            raise ValueError("x_0 should be (lat, lon, battery) and x_T (lat, lon)")

        # check t_0
        if hindcast_source['data_source_type'] == 'analytical_function':
            t_0 = datetime.fromtimestamp(t_0, timezone.utc)
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
        # Plan on GT
        self.plan_on_gt = plan_on_gt

        # Initialize the data_source class variables from the inputs provided
        self.hindcast_data_source = self.update_data_source(hindcast_source)
        if not self.plan_on_gt:
            self.forecast_data_source = self.update_data_source(forecast_source, forecast=True)

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
        # About the task
        Nav_string = "Navigate from {} at time {} to {}.".format(
            self.x_0[:3], datetime.fromtimestamp(self.x_0[3], timezone.utc), self.x_T)

        # What is used for Simulation
        if self.hindcast_data_source['data_source_type'] != 'analytical_function':
            Sim_string = "Simulate with GT current files from {} to {}".format(
                self.hindcast_data_source['grid_dict']['t_range'][0],
                self.hindcast_data_source['grid_dict']['t_range'][1])
        else:
            Sim_string = "Simulating with analytical current function."

        # What is used for Planning
        if self.plan_on_gt:
            Plan_string = "Planning on GT."
        else:
            if self.forecast_data_source['data_source_type'] == 'single_nc_file':
                Plan_string = "Planning on {} Forecast files starting from {} to {}".format(
                    len(self.forecast_data_source['content']),
                    self.forecast_data_source['t_forecast_coverage'][0],
                    self.forecast_data_source['t_forecast_coverage'][1])
            else:
                Plan_string = "Planning on analytical forecast."
        return Nav_string + '\n' + Sim_string + '\n' + Plan_string

    def update_data_source(self, data_source, forecast=False):
        """Set the data_source class variables for the different source type inputs.
        Inputs:
            hindcast_source/forecast_source     see description of init.
        """

        # Derive the Data Sources
        if data_source['data_source_type'] in ['single_nc_file', 'multiple_daily_nc_files']:
            # Step 1: get the file dict
            file_dict = self.get_file_dicts(data_source['data_source'])
            # Step 2: create the data_source dict
            updated_data_source = {'data_source_type': data_source['data_source_type'],
                                   'content': file_dict}
            # Step 2: add grid dict to the data_source object
            if not forecast:
                updated_data_source['grid_dict'] = self.derive_grid_dict_from_files(updated_data_source)
            else: # need to save more stuff in the grid_dict
                updated_data_source['t_forecast_coverage'] = [
                    updated_data_source['content'][0]['t_range'][0],  # t_0 of fist forecast file
                    updated_data_source['content'][-1]['t_range'][1]] # t_final of last forecast file
                cur_for_idx = self.get_current_forecast_idx(updated_data_source['content'])
                updated_data_source['current_forecast_idx'] = cur_for_idx
                updated_data_source['grid_dict'] = self.derive_grid_dict_from_files(updated_data_source)

        elif data_source['data_source_type'] == 'cop_opendap':
            # Step 1: initialize the session with the provided credentials
            data_store = utils.simulation_utils.copernicusmarine_datastore(data_source['data_source']['DATASET_ID'],
                                                                           data_source['data_source']['USERNAME'],
                                                                           data_source['data_source']['PASSWORD'])
            # Step 1.2: subset to only the variables we care about
            DS_currents = xr.open_dataset(data_store)[['uo', 'vo']].isel(depth=0)
            # Step 3: create the data_source dict
            updated_data_source = {'data_source_type': data_source['data_source_type'],
                                   'content': DS_currents,
                                   'grid_dict': self.derive_grid_dict_from_xarray(DS_currents)}

        elif data_source['data_source_type'] == 'analytical_function':
            # Step 0: check if it's a subclass of AnalyticalField
            if not issubclass(type(data_source['data_source']), utils.AnalyticalField):
                raise ValueError("For analytical_function we need 'data_source' to be subclass of utils.AnalyticalField")

            updated_data_source = {'data_source_type': data_source['data_source_type'],
                                         'content': data_source['data_source'],
                                         'grid_dict': data_source['data_source'].get_grid_dict()[0]}
            # modify to be datetime objects
            updated_data_source['grid_dict']['t_range'] = [datetime.fromtimestamp(rel_time[0], timezone.utc) for
                                                                 rel_time in updated_data_source['grid_dict']['t_range']]
        else:
            raise ValueError("data_source[\'data_source_type\'] must be in {\"single_nc_file\","
                             " \"multiple_daily_nc_files\", \"cop_opendap\", \"analytical_function\"}")

        return updated_data_source

    def viz(self, time=None, video=False, filename=None, cut_out_in_deg=0.8, add_ax_func=None,
            html_render=None, temp_horizon_viz_in_h=120, return_ax=False, plot_start_target=True,
            temporal_stride=1, time_interval_between_pics=200, hours_to_hj_solve_timescale=3600):
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
        - hours_to_hj_solve_timescale: factor to multiply hours with to get to current field temporal units.

        Returns:
            None
        """
        print("Note only the GT hindcast data is currently visualized")
        # Step 0: Find the time, lat, lon bounds for data_subsetting
        t_interval, lat_interval, lon_interval = utils.simulation_utils.convert_to_lat_lon_time_bounds(self.x_0, self.x_T,
                                                                                deg_around_x0_xT_box=cut_out_in_deg,
                                                                                temp_horizon_in_h=temp_horizon_viz_in_h,
                                                                                hours_to_hj_solve_timescale=hours_to_hj_solve_timescale)

        if add_ax_func is None:
            def add_ax_func(ax, time=None, x_0=self.x_0[:2], x_T=self.x_T[:2]):
                del time
                if plot_start_target:
                    ax.scatter(x_0[0], x_0[1], c='r', marker='o', s=200, label='start')
                    goal_circle = plt.Circle((x_T[0], x_T[1]), self.x_T_radius, color='g', fill=True, alpha=0.6, label='target')
                    ax.add_patch(goal_circle)
                    plt.legend(loc='upper right')

        # if we want to visualize with video
        if time is None and video:
            # Step 1: get the data_subset for plotting
            grids_dict, u_data, v_data = utils.simulation_utils.get_current_data_subset(t_interval, lat_interval, lon_interval,
                                                                 data_source=self.hindcast_data_source,
                                                                 max_temp_in_h=120)
            # create animation with extra func
            utils.plotting_utils.viz_current_animation(grids_dict['t_grid'][::temporal_stride],
                                             grids_dict, u_data, v_data,
                                             interval=time_interval_between_pics,
                                             ax_adding_func=add_ax_func, html_render=html_render,
                                             save_as_filename=filename)
        # otherwise plot static image
        else:
            # Step 1: get the data_subset for plotting
            if time is None:
                time = datetime.fromtimestamp(self.x_0[3], tz=timezone.utc)
            t_interval_static = [time - timedelta(hours=8), time + timedelta(hours=8)]
            grids_dict, u_data, v_data = utils.simulation_utils.get_current_data_subset(t_interval_static, lat_interval, lon_interval,
                                                                 data_source=self.hindcast_data_source)
            # plot underlying currents at time
            ax = utils.plotting_utils.visualize_currents(time.timestamp(), grids_dict, u_data, v_data,
                                               # figsize=figsize,
                                               autoscale=True, plot=False)
            # add the start and goal position to the plot
            add_ax_func(ax)
            if return_ax:
                return ax
            else:
                plt.show()

    def is_on_land(self, point):
        """Returns True/False if the closest grid_point to the self.cur_state is on_land."""
        # get idx of closest grid-points
        x_idx = (np.abs(self.hindcast_data_source['grid_dict']['x_grid'] - point[0])).argmin()
        y_idx = (np.abs(self.hindcast_data_source['grid_dict']['y_grid'] - point[1])).argmin()
        # Note: the spatial_land_mask is an array with [Y, X]
        return self.hindcast_data_source['grid_dict']['spatial_land_mask'][y_idx, x_idx]

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
            grid_dict = utils.simulation_utils.get_grid_dict_from_file(file)
            # append the file to it:
            grid_dict['file'] = file
            list_of_dicts.append(grid_dict)
        # sort the list
        list_of_dicts.sort(key=lambda dict: dict['t_range'][0])
        return list_of_dicts

    @staticmethod
    def derive_grid_dict_from_files(data_source):
        """Helper function to create the a grid dict from one or multiple files in self.hindcast_data_source."""

        # check if forecast
        if 'current_forecast_idx' in data_source:
            idx_file = data_source['current_forecast_idx']
        else:
            idx_file = 0

        # Step 0: Checks before extracting the data
        file_source = data_source['data_source_type'] in ['single_nc_file', 'multiple_daily_nc_files']
        if not file_source:
            raise ValueError("Wrong data_source_type for function derive_grid_dict_from_files. Only for nc files.")
        if len(data_source['content']) == 0:
            raise ValueError("No nc files found in the data_source_type folder.")

        # get spatial coverage by reading in the first file
        f = xr.open_dataset(data_source['content'][idx_file]['file'])
        xgrid = f.variables['lon'].data
        ygrid = f.variables['lat'].data

        y_range = [ygrid[0], ygrid[-1]]
        x_range = [xgrid[0], xgrid[-1]]

        # get time_range
        if data_source['data_source_type'] == 'single_nc_file':
            time_range = data_source['content'][idx_file]['t_range']
        elif data_source['data_source_type'] == 'multiple_daily_nc_files':
            # get time-range by iterating over the return elements that are consecutive/exactly 1h apart
            for idx in range(len(data_source['content']) - 1):
                if data_source['content'][idx]['t_range'][1] + timedelta(hours=1) \
                        == data_source['content'][idx + 1]['t_range'][0]:
                    continue
                else:
                    break
            time_range = [data_source['content'][0]['t_range'][0], data_source['content'][idx + 1]['t_range'][1]]

        # get the land mask
        u_data = f.variables['water_u'].data[:, 0, :, :]
        # adds a mask where there's a nan (on-land)
        u_data = np.ma.masked_invalid(u_data)

        return {"t_range": time_range, "y_range": y_range, "x_range": x_range,
                'spatial_land_mask':u_data[0, :, :].mask, 'x_grid': xgrid, 'y_grid': ygrid}

    @staticmethod
    def derive_grid_dict_from_xarray(DS_currents):
        """Helper function to create the grid_dict from a opendap object"""
        time_range = [
            datetime.fromtimestamp(DS_currents.variables['time'][0].data.astype(int) * 1e-9, tz=timezone.utc),
            datetime.fromtimestamp(DS_currents.variables['time'][-1].data.astype(int) * 1e-9, tz=timezone.utc)]
        y_grid = DS_currents.variables['latitude'].data
        x_grid = DS_currents.variables['longitude'].data
        grid_dict = {"t_range": time_range,
                     "y_range": [y_grid[0], y_grid[-1]], "x_range": [x_grid[0], x_grid[-1]],
                     'spatial_land_mask': False, 'x_grid': x_grid, 'y_grid': y_grid}

        return grid_dict

    def check_data_compatibility(self, t, points):
        """Check if given forecast and hindcasts contain all points at time t."""
        self.check_hindcast_compatibility(t, points)
        if not self.plan_on_gt:
            self.check_forecast_comatibility(t, points)

    def check_hindcast_compatibility(self, t, points):
        """Helper function to check if the Hindcast files cover all points at t_0.
        Input:
            t       datetime_object of time
            points  list of [lon, lat] points
        """
        # Step 1: check if t_0 is in the Hindcast time-range
        if not (self.hindcast_data_source['grid_dict']['t_range'][0] < t < self.hindcast_data_source['grid_dict']['t_range'][1]):
            raise ValueError("Hindcast files do not cover {}.".format(t))
        # Step 2: check if x_0 and x_T are in the spatial coverage of the Hindcast files
        for point in points:
            if not (self.hindcast_data_source['grid_dict']['x_range'][0] < point[0] < self.hindcast_data_source['grid_dict']['x_range'][1]):
                raise ValueError("Hindcast files does not contain {} in longitude range.".format(point))
            if not (self.hindcast_data_source['grid_dict']['y_range'][0] < point[1] < self.hindcast_data_source['grid_dict']['y_range'][1]):
                raise ValueError("Hindcast files does not contain {} in latitude range.".format(point))

    def check_forecast_comatibility(self, t, points):
        """Helper function to check if the most recent Forecast file covers all points at t_0.
        Input:
            t       datetime_object of time
            points  list of [lon, lat] points
        """

        if self.forecast_data_source['data_source_type'] == 'single_nc_file':
            t_range = self.forecast_data_source['content'][self.forecast_data_source['current_forecast_idx']]['t_range']
            x_range = self.forecast_data_source['content'][self.forecast_data_source['current_forecast_idx']]['x_range']
            y_range = self.forecast_data_source['content'][self.forecast_data_source['current_forecast_idx']]['y_range']
        else:
            t_range = self.forecast_data_source['grid_dict']['t_range']
            x_range = self.forecast_data_source['grid_dict']['x_range']
            y_range = self.forecast_data_source['grid_dict']['y_range']

        # Step 1: check if at t_0 there's a forecast available.
        if not (t_range[0] < t < t_range[1]):
            raise ValueError("Most recent Forecast file does not cover {}.".format(t))
        # Step 2: check if x_0 and x_T are in the spatial coverage of the most recent Forecast file
        for point in points:
            if not (x_range[0] < point[0] < x_range[1]):
                raise ValueError("Most recent Forecast file does not contain {} in longitude range.".format(point))
            if not (y_range[0] < point[1] < y_range[1]):
                raise ValueError("Most recent Forecast file does not contain {} in latitude range.".format(point))

    def get_current_forecast_idx(self, forecasts_files_dicts):
        """Get the index of the most recent forecast available t_0."""
        # Filter on the list to get all files where t_0 is contained.
        dics_containing_t_0 = list(
            filter(lambda dic: dic['t_range'][0] < self.t_0 < dic['t_range'][1], forecasts_files_dicts))
        # Basic Sanity Check if this list is empty no file contains t_0
        if len(dics_containing_t_0) == 0:
            raise ValueError("None of the forecast files contains t_0.")
        # As the dict is time-ordered we simple need to find the idx of the last one in the dics_containing_t_0
        for idx, dic in enumerate(forecasts_files_dicts):
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
