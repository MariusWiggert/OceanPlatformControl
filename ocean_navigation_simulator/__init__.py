# Make sub-folders visible to the module
from ocean_navigation_simulator import (
    controllers,
    environment,
    generative_error_model,
    ocean_observer,
    problem_factories,
    utils,
)

# specify version (for pip installation)
__version__ = "0.1.0"
__all__ = (
    "controllers",
    "environment",
    "ocean_observer",
    "generative_error_model",
    "problem_factories",
    "utils",
)
