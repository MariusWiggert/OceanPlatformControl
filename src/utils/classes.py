import parcels as p
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, Field, Variable
import glob, imageio
from utils import kernels, particles
from datetime import datetime,timedelta

class Position:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

class ProblemSet:
    def __init__(self, fieldset):
        self.fieldset = fieldset

    def create_problem(self):
        """ TODO: Randomly generate Problems with valid x_0 and x_T """

class Problem:
    def __init__(self, fieldset, x_0, x_T):
        self.fieldset = fieldset                        
        self.x_0 = x_0                                  # Position
        self.x_T = x_T                                  # Position

class Planner:
    """ All Planners should inherit this class """

    def step_forward(self, observation):
        """ TODO: returns (thrust, header) for the next timestep """
        raise NotImplementedError()

class TrajectoryTrackingController:
    pass 

class Simulator:
    """
    Functions as our "Parcels". Users will:
    1. Create Simulator object with planner, problem, 
    2. Execute simulator.run()
    3. Fetch trajectory plot with plot_trajectory
    4. Evaluate planner solving given problem under some metric
    """
    def __init__(self, planner, problem, settings):
        self.planner = planner             # list of Planner
        self.problem = problem            
        self.current_state = problem.x_0
        self.time_origin = problem.fieldset.field.time_origin
        self.delta_time = 0
        self.trajectory = []
        self.settings = settings            # {'project_dir', 'dt', 'T', ...}

    def run(self):
        dt, T = self.settings['dt'], self.settings['T']
        for t in range(0, T, dt):
            """ TODO: implement the following parts """
            # self.planner.step_forward(...)
            # update self.trajectory
            continue

    def plot_trajectory(self, plotting_type = 'gif'):
        """ Captures the whole trajectory - energy, position, etc over time
        Accesses the trajectory and fieldset from the problem.
        """
        for i, x in enumerate(self.trajectory):
            # under the assumption that x is a Position
            pset = p.ParticleSet.from_list(
                    fieldset=self.problem.fieldset,  # the fields on which the particles are advected
                    pclass=particles.ScipyParticle, # the type of particles (JITParticle or ScipyParticle)
                    lon=[x.lon],  # a vector of release longitudes
                    lat=[x.lat],  # a vector of release latitudes
                )

            pset.show(savefile=self.settings['project_dir'] + '/viz/pics_2_gif/particles' + str(i).zfill(2), field='vector', land=True,
                vmax=1.0, show_time=0.)
    
    def evaluate(self, planner, problem):
        """ TODO: Evaluate the planner on the given problem by some metrics """