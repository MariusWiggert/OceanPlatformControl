# Make functions from different files directly available under utils.analytical_fields
from .AnalyticalField import AnalyticalField
from .PeriodicDoubleGyre import PeriodicDoubleGyre
from .HighwayCurrent import FixedCurrentHighwayField

__all__ = ("AnalyticalField", "PeriodicDoubleGyre", "FixedCurrentHighwayField")