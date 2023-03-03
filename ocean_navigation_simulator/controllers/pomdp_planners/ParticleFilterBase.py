class ParticleFilterBase:
	def __init__(self) -> None:
		return
	
	def dynamics_update(self) -> None:
		raise Exception("Not Implemented: Dynamics Update")

	def observation_update(self) -> None:
		raise Exception("Not Implemented: Observation Update")
	
	def full_bayesian_update(self) -> None:
		raise Exception("Not Implemented: Full Bayesian Update")