from ocean_navigation_simulator.utils import units

import pytest


def test__seconds_to_hours():
    assert units.seconds_to_hours(3600) == 1
