from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory

# from ocean_navigation_simulator.utils import cluster_utils


ArenaFactory.analyze_files("Copernicus", "forecast", "GOM", True)
ArenaFactory.analyze_files("HYCOM", "hindcast", "GOM", True)