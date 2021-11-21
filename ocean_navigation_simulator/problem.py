import yaml, os, netCDF4, abc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# import from other utils
import ocean_navigation_simulator.utils.plotting_utils as plot_utils
from ocean_navigation_simulator.utils.simulation_utils import convert_to_lat_lon_time_bounds, get_current_data_subset
from ocean_navigation_simulator.utils.simulation_utils import get_abs_time_grid_for_hycom_file


# Basis function for the problem class
class BaseProblem(metaclass=abc.ABCMeta):
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

        # TO IMPLEMENT USAGE/FOR FUTURE
        forecast_delay_in_h:
            The hours of delay when a forecast becomes available
            e.g. forecast starts at 1st of Jan but only available from HYCOM 48h later on 3rd of January
        noise:
            # TODO: optionally implement a way to add noise to the hindcasts

        # TO REVIEW/THINK ABOUT
        x_t_tol:
            Radius around x_T that when reached counts as "target reached"
            # Note: not used currently as the sim config has that value too.
    """

    def __init__(self, x_0, x_T, t_0, platform_config_dict, plan_on_gt=False,
                 forecast_delay_in_h=0., noise=None, x_t_tol=0.1):

        # Plan on GT
        self.plan_on_gt = plan_on_gt

        # Need to be derived in the child_classes
        self.hindcast_grid_dict = None
        self.forecasts_dicts = None
        self.data_access = None
        self.local_hindcast_file = None

        # Basic check of inputs
        if len(x_0) != 3 or len(x_T) != 2:
            raise ValueError("x_0 should be (lat, lon, battery) and x_T (lat, lon)")

        # Log start, goal and forecast delay
        self.x_0 = x_0 + [t_0.timestamp()]
        self.x_T = x_T
        self.forecast_delay_in_h = forecast_delay_in_h

        # derive relative batter dynamics variables from config_dict
        # The if-clause is in case we want to specify it as path to a yaml
        if isinstance(platform_config_dict, str):
            # get the local project directory (assuming a specific folder structure)
            project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # load simulator sim_settings YAML
            with open(project_dir + '/configs/' + platform_config_dict) as f:
                platform_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.dyn_dict = self.derive_platform_dynamics(platform_config_dict)

    def __repr__(self):
        """Returns the string representation of a Problem, to be used for debugging.

        Returns:
            A String
        """
        # Print problem specs
        print("Navigate from {} at time {} to {}.".format(
            self.x_0[:3], datetime.utcfromtimestamp(self.x_0[3]), self.x_T
        ))
        print("Simulate with GT current files from {} to {}".format(
            self.hindcast_grid_dict['gt_t_range'][0],
            self.hindcast_grid_dict['gt_t_range'][1]))
        if self.plan_on_gt:
            print("Planning on GT.")
        else:
            print("Planning on {} Forecast files starting from {} to {}".format(
                len(self.forecasts_dicts),
                self.forecasts_dicts[0]['t_range'][0],
                self.forecasts_dicts[-1]['t_range'][0]))

        return ""

    def viz(self, time=None, video=False, filename=None, cut_out_in_deg=0.8,
            html_render=None, temp_horizon_viz_in_h=None):
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

        print("Note only the GT file is currently visualized")
        # Step 1: get the data_subset for plotting (flexible query can be with local or C3 file)
        grids_dict, u_data, v_data = get_current_data_subset(t_interval, lat_interval, lon_interval,
                                                             data_type='H', access=self.data_access,
                                                             file=self.local_hindcast_file)

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

    def get_hindcast_grid_dict(self):
        return None

    @abc.abstractmethod
    def get_forecast_dicts(self, forecast_folder):
        """ Creates an list of dicts ordered according to time available, one for each forecast file available.
        The dicts for each forecast file contains: {'t_range': [<datetime object>, T], 'file': <filepath>})
        """

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


# Problem for local files
class Problem(BaseProblem):
    """Problem Class if the current files are available locally.
    Input variables are same as in the Base class except two new ones:

    hindcast_file           absolute path to hindcast file
    forecast_folder         absolute path to folder containing all forecast files
    """
    def __init__(self, x_0, x_T, t_0, platform_config_dict, hindcast_file, forecast_folder=None,
                 plan_on_gt=False, forecast_delay_in_h=0., noise=None, x_t_tol=0.1):

        # initialize the Base Problem
        super().__init__(x_0, x_T, t_0, platform_config_dict, plan_on_gt, forecast_delay_in_h, noise, x_t_tol)

        # Derive the hindcast_grid_dict
        self.local_hindcast_file = hindcast_file
        self.hindcast_grid_dict = self.get_grid_dict_from_file(hindcast_file)

        # Derive the list forecasts_dicts
        if not self.plan_on_gt:
            self.forecasts_dicts = self.get_forecast_dicts(forecast_folder)

        # log variables for data_acces
        self.data_access = 'local'

    def get_grid_dict_from_file(self, file, data_type='H'):
        """Helper function to create a grid dict from a local file.
        Input: hindcast_file
        """
        f = netCDF4.Dataset(file)
        # get the time coverage in POSIX
        t_grid = get_abs_time_grid_for_hycom_file(f, data_type=data_type)
        # create dict
        return {"gt_t_range": [datetime.utcfromtimestamp(t_grid[0]), datetime.utcfromtimestamp(t_grid[-1])],
                "gt_y_range": [f.variables['lat'][:][0], f.variables['lat'][:][-1]],
                "gt_x_range": [f.variables['lon'][:][0], f.variables['lon'][:][-1]]}

    def get_forecast_dicts(self, forecast_folder):
        """Helper function to create a list of dicts of the available forecasts from the local folder.
        Starting from the most recent at t_0 in the future."""

        # get a list of files from the folder
        forecast_files_list = [forecast_folder + f for f in os.listdir(forecast_folder) if
                               (os.path.isfile(os.path.join(forecast_folder, f)) and f != '.DS_Store')]

        # iterate over all files to extract the t_ranges and put them in an ordered list of dicts
        forecast_dicts = []
        for file in forecast_files_list:
            grid_dict = self.get_grid_dict_from_file(file, data_type='F')
            forecast_dicts.append({'t_range': grid_dict['gt_t_range'], 'file': file})
        # sort the list
        forecast_dicts.sort(key=lambda dict: dict['t_range'][0])

        return forecast_dicts


# Problem for C3 based data
class C3Problem(BaseProblem):
    """Problem Class if the current files are taken from the C3 database.
        Input variables are same as in the Base class.
        """
    def __init__(self, x_0, x_T, t_0, platform_config_dict, plan_on_gt=False,
                 forecast_delay_in_h=0., noise=None, x_t_tol=0.1):

        # initialize the Base Problem
        super().__init__(x_0, x_T, t_0, platform_config_dict, plan_on_gt, forecast_delay_in_h, noise, x_t_tol)

        # Derive the hindcast_grid_dict and the list forecast_dicts
        self.hindcast_grid_dict = self.get_hindcast_grid_dict()
        if not self.plan_on_gt:
            self.forecasts_dicts = self.get_forecast_dicts()

        # log variables for data_acces
        self.data_access = 'C3'

    def get_hindcast_grid_dict(self):
        """Helper function to create the hindcast grid dict from the multiple files in the C3 DB.
        The idea is: how many consecutive daily hindcast files do we have starting from t_0.
        """
        # Step 1: get required file references and data from C3 file DB
        # Step 1.1: Getting time and formatting for the db query
        start = datetime.utcfromtimestamp(self.x_0[3])

        # Step 1.2: Getting correct range of nc files from database
        filter_string = 'start>=' + '"' + start.strftime("%Y-%m-%d") + '"' + \
                        ' && status==' + '"' + 'downloaded' + '"'
        objs_list = c3.HindcastFile.fetch({'filter': filter_string, "order": "start"}).objs

        # some basic sanity checks
        if objs_list is None:
            raise ValueError("No files in the database for and after the selected start_time")

        # get spatial coverage
        y_range = [objs_list[0].subsetOptions.geospatialCoverage.start.latitude,
                   objs_list[0].subsetOptions.geospatialCoverage.end.latitude]
        x_range = [objs_list[0].subsetOptions.geospatialCoverage.start.longitude,
                   objs_list[0].subsetOptions.geospatialCoverage.end.longitude]

        # get time_range by iterating over the return elements that are consecutive/exactly 1h apart
        starts_list = [obj.start for obj in objs_list]
        ends_list = [obj.end for obj in objs_list]
        for idx in range(len(starts_list) - 1):
            if ends_list[idx] + timedelta(hours=1) == starts_list[idx + 1]:
                continue
            else:
                break
        time_range = [starts_list[0], ends_list[idx + 1]]

        # create and return dict
        return {"gt_t_range": time_range, "gt_y_range": y_range, "gt_x_range": x_range}

    def get_forecast_dicts(self, forecast_folder=None):
        """Helper function to create a list of dicts of the available forecasts from the C3 DB.
        Starting from the most recent at t_0 in the future."""

        # Step 1: get required file references and data from C3 file DB
        # get all relevant forecasts including most recent at t_0 and all the ones after
        from_run_onwards = datetime.utcfromtimestamp(self.x_0[3]) - timedelta(days=1)
        filter_string = 'runDate>=' + '"' + from_run_onwards.strftime("%Y-%m-%dT%H:%M:%S") + '"'
        objs_list = c3.HycomFMRC.fetch(spec={'include': "[this, fmrcFiles.file]",
                                             'filter': filter_string,
                                             "order": "runDate"}
                                       ).objs

        # basic sanity check
        if objs_list is None:
            raise ValueError("No forecast runs in the database for and after the selected start_time")

        # Step 2: create a list of dicts with one dict for each run/forecast file
        forecast_dicts = []
        for run in objs_list:
            t_range = [run.timeCoverage.start, run.timeCoverage.end]
            forecast_dicts.append({'t_range': t_range, 'file': run.fmrcFiles[0].file.url})

        # sorting after t_range start (doubling because already in db query but to be safe)
        forecast_dicts.sort(key=lambda dict: dict['t_range'][0])

        return forecast_dicts
