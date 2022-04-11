class Problem:
    """
    A path planning problem for a Planner to solve.

    Attributes:
        start_state:
            PlatformState object specifying where the agent will start
        end_region:
            TODO: [UndefinedObject] specifying the region the agent should try reach (positive reward)
        obstacle_regions:
            TODO: [UndefinedObject] specifying the region(s) the agent should avoid (negative reward)
        config:
            A dict specifying the platform/problem parameters
            TODO: make this config into class?

    """

    def __init__(self, start_state, end_region, obstacle_regions, config):
        self.start_state = start_state
        self.end_region = end_region
        self.obstacle_regions = obstacle_regions
        self.config = config

    # TODO: add functions from old problem.py
