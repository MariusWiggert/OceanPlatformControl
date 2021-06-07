import yaml
import parcels as p
import math
import numpy as np
import glob, os, imageio
from src.utils import hycom_utils
from os import listdir
from os.path import isfile, join


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        x_0:
            The starting state, represented as (lon, lat, battery_level, time).
            Note that time is implemented as relative time right now (hence always x_0[3]=0)
            # TODO: implement possibility to put in datetime objects as x_0
        x_T:
            The target state, represented as (lon, lat).
            # TODO: currently we do point-2-point navigation though ultimately we'd like
            this to be a set representation (point-2-region) because that is the more general formulation.
        t_0:
            A np.datetime object of the absolute starting time
            # TODO: currently this doesn't have an effect
            we assume starting at 0 time of both the forecasts and hindcast file
        # TODO: implement flexible loading of the nc4 files outside of here
        hindcast_file:
            Path to the nc4 files of forecasted ocean currents. Will be used as true currents (potentially adding noise)
        forecast_folder:
            Path to the nc4 files of forecasted ocean currents
        noise:
            # TODO: optionally implement a way to add noise to the hindcasts
        x_t_tol:
            Radius around x_T that when reached counts as "target reached"
        config_yaml:
            A YAML file for the platform configurations.
        fixed_time_index:
            An index of a fixed time index from the hindcast file.
            applicable when the time is fixed, i.e. the currents are not time varying.
            # TODO: currently not implemented, easier when taken care of outside of this.
        project dir:
            Only needed if the data is stored outside the repo
    """

    def __init__(self, x_0, x_T, t_0, hindcast_file, forecast_folder=None, noise=None, x_t_tol=0.1, config_yaml='platform.yaml',
                 fixed_time_index=None, project_dir=None):

        if project_dir is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # load the respective fieldsets
        self.hindcast_file = hindcast_file  # because the simulator loads only subsetting of it
        self.hindcast_fieldset = hycom_utils.get_hycom_fieldset(hindcast_file)  # this loads in all, good for plotting
        self.forecast_list = [forecast_folder + f for f in listdir(forecast_folder) if isfile(join(forecast_folder, f))]
        self.forecast_list.sort()

        if fixed_time_index is not None:
            self.fixed_time_index = fixed_time_index
            print("Fixed-time fieldset currently not implemented")
            raise NotImplementedError()
        else:  # variable time
            self.fixed_time_index = None
            print("GT fieldset from  {} to {}".format(self.hindcast_fieldset.time_origin,
                                                      self.hindcast_fieldset.gridset.grids[0].timeslices[0][-1]))
            print("GT Resolution of {} h".format(math.ceil(self.hindcast_fieldset.gridset.grids[0].time[1] / 3600)))
            print("Forecast files from \n{} \n to \n{}".format(self.forecast_list[0],self.forecast_list[-1]))

        if len(x_0) == 2:  # add 100% charge and time
            x_0 = x_0 + [1., 0.]
        elif len(x_0) == 3:  # add time only
            x_0.append(0.)

        self.x_0 = x_0
        self.x_T = x_T
        self.t_0 = t_0
        self.project_dir = project_dir
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

        if self.fixed_time_index is None and time is None and gif:
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
        elif self.fixed_time_index is None and time is None and not gif:
            pset.show(field='vector', show_time=0)
        elif time is None:  # fixed time index
            pset.show(field='vector', show_time=self.fixed_time_index)
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


class WaypointTrackingProblem(Problem):
    #TODO: not fit to new closed loop controller yet
    """ Only difference is the added waypoints to the problem """

    def __init__(self, real_fieldset, forecasted_fieldset, x_0, x_T, project_dir, waypoints,
                 config_yaml='platform.yaml',
                 fixed_time_index=None):
        super().__init__(real_fieldset=real_fieldset,
                         forecasted_fieldset=forecasted_fieldset,
                         x_0=x_0,
                         x_T=x_T,
                         project_dir=project_dir,
                         config_yaml=config_yaml,
                         fixed_time_index=fixed_time_index)
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
