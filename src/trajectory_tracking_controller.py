class TrajectoryTrackingController:
    """
    TODO: implement controller to go along planned trajectory for platforms.
    """
    def __init(self, trajectory, state):  # Will need planned trajectory and current state
        # 0) Set up the variables
        _L1_damping = 0.75
        _L1_period = 17  # we should set this to dt.
        dist_min = '?'
        groundSpeed = "our current speed"  # shouldn't be needed, i.e. doesn't matter

        # 1) calculate the L1_Gain
        K_L1 = 4.0 * _L1_damping * _L1_damping

        # 2) Calculate the L1_dist
        _L1_dist = max(0.3183099 * _L1_damping * _L1_period * groundSpeed, dist_min);
