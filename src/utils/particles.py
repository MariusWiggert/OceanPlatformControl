import numpy as np
import parcels as p
# define custom particle that has extra Variables such as the goal location, and control trajectory


class TargetParticle(p.JITParticle):
	# target
	lat_target = p.Variable('lat_target', dtype=np.float32, initial=0., to_write=False)
	lon_target = p.Variable('lon_target', dtype=np.float32, initial=0., to_write=False)
	v_max = p.Variable('v_max', dtype=np.float32, initial=0., to_write=False)


class OpenLoopParticle(p.ScipyParticle):
	# target and control trajectory
	control_traj = p.Variable('control_traj', initial=0., to_write=False, dtype=object)     # 2xN long
	control_time = p.Variable('control_time', initial=0., to_write=False, dtype=object)     # 1xN+1 long

	v_max = p.Variable('v_max', dtype=np.float32, initial=1., to_write=False)

#%%
