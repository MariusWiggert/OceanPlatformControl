import yaml
import parcels as p


class Problem:
    """A path planning problem for a Planner to solve.

    Attributes:
        fieldset:
            The fieldset contains data about ocean currents and is more rigorously defined in the
            parcels documentation.
        x_0:
            The starting state, represented as (lon, lat, battery_level, time).
        x_T:
            The target state, represented as (lon, lat).
        project_dir:
            A string giving the path to the project directory.
        config_yaml:
            A YAML file for the Problem configurations.
        fixed_time_index:
            An index of a fixed time. Note that the fixed_time_index is only
            applicable when the time is fixed, i.e. the currents are not time varying.
    """

    def __init__(self, fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None):
        self.fieldset = fieldset
        if fieldset.U.grid.time.shape[0] == 1:  # check if we do a fixed or variable current problem
            self.fixed_time_index = 0
        else:
            self.fixed_time_index = fixed_time_index
        if len(x_0) == 2:  # add 100% charge and time
            x_0 = x_0 + [1., 0.]
        elif len(x_0) == 3:  # add time only
            x_0.append(0.)
        self.x_0 = x_0
        self.x_T = x_T
        self.dyn_dict = self.derive_platform_dynamics(project_dir, config=config_yaml)

    def __repr__(self):
        """Returns the string representation of a Problem, to be used for debugging.

        Returns:
            A String
        """
        return "Problem(x_0: {0}, x_T: {1})".format(self.x_0, self.x_T)

    def viz(self):
        """Visualizes the problem in a plot.

        Returns:
            None
        """
        pset = p.ParticleSet.from_list(fieldset=self.fieldset,  # the fields on which the particles are advected
                                       pclass=p.ScipyParticle,  # the type of particles (JITParticle or ScipyParticle)
                                       lon=[self.x_0[0], self.x_T[0]],  # a vector of release longitudes
                                       lat=[self.x_0[1], self.x_T[1]],  # a vector of release latitudes
                                       )
        pset.show(field='vector')

    def derive_platform_dynamics(self, project_dir, config):
        """Derives battery capacity dynamics for the specific dt.

        Args:
            project_dir:
                See class docstring
            config:
                See config_yaml in class docstring

        Returns:
            A dictionary of settings for the Problem, i.e. {'charge': __, 'energy': __, 'u_max': __}
        """

        # load in YAML
        with open(project_dir + '/config/' + config) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            platform_specs = config_data['platform_config']

        # derive calculation
        cap_in_joule = platform_specs['battery_cap'] * 3600
        energy_coeff = (platform_specs['drag_factor'] * (1 / platform_specs['motor_efficiency'])) / cap_in_joule
        charge_factor = platform_specs['avg_solar_power'] / cap_in_joule
        platform_dict = {'charge': charge_factor, 'energy': energy_coeff, 'u_max': platform_specs['u_max']}

        return platform_dict
