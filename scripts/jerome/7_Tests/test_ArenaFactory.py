import pandas as pd
import os

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.FileProblemFactory import FileProblemFactory

factory = FileProblemFactory()
problem = factory.next_problem(skip=100)

files = ArenaFactory.download_hycom_forecast(problem)
os.system('ls -la /tmp/hycom_forecasts')