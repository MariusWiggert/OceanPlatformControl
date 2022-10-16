# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors. Adapted by Marius Wiggert and Jerome Jeannine.

"""Common unit conversion functions and classes."""

import datetime as datetime
import typing
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import contextlib

_METERS_PER_FOOT = 0.3048
_METERS_PER_DEG_LAT_LON = 111120


@contextlib.contextmanager
def timing(string, verbose: Optional[int] = 0):
    """ Simple tool to check how long a specific code-part takes."""
    if verbose > 0:
        start = time.time()
    yield
    if verbose > 0:
        print(string.format(time.time()-start))


class Distance:
    """A compact distance unit."""

    def __init__(self,
                 *,
                 m: float = 0.0,
                 meters: float = 0.0,
                 km: float = 0.0,
                 kilometers: float = 0.0,
                 feet: float = 0.0,
                 deg: float = 0.0
                 ):
        # Note: distance is stored as degree (because that's how it is used almost always)
        self._distance = (m + meters + (km + kilometers) * 1000.0 + feet * _METERS_PER_FOOT) / _METERS_PER_DEG_LAT_LON + deg

    @property
    def m(self) -> float:
        """Gets distance in meters."""
        return self._distance * _METERS_PER_DEG_LAT_LON

    @property
    def meters(self) -> float:
        """Gets distance in meters."""
        return self.m * _METERS_PER_DEG_LAT_LON

    @property
    def deg(self) -> float:
        """Gets distance in degree."""
        return self._distance

    @property
    def km(self) -> float:
        """Gets distance in kilometers."""
        return self._distance * _METERS_PER_DEG_LAT_LON / 1000.0

    @property
    def kilometers(self) -> float:
        """Gets distance in kilometers."""
        return self.km

    @property
    def feet(self) -> float:
        return self._distance * _METERS_PER_DEG_LAT_LON / _METERS_PER_FOOT

    def __add__(self, other: 'Distance') -> 'Distance':
        if isinstance(other, Distance):
            return Distance(deg=self.deg + other.deg)
        else:
            raise NotImplementedError(f'Cannot add Distance and {type(other)}')

    def __sub__(self, other: 'Distance') -> 'Distance':
        if isinstance(other, Distance):
            return Distance(deg=self.deg - other.deg)
        else:
            raise NotImplementedError(f'Cannot subtract Distance and {type(other)}')

    @typing.overload
    def __truediv__(self, other: float) -> 'Distance':
        ...

    @typing.overload
    def __truediv__(self, other: datetime.timedelta) -> 'Velocity':
        # velocity = change in distance / change in time.
        ...

    @typing.overload
    def __truediv__(self, other: 'Distance') -> float:
        ...

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Distance(deg=self.deg / other)
        elif isinstance(other, datetime.timedelta):
            return Velocity(mps=self.m / other.total_seconds())
        elif isinstance(other, Distance):
            return self.deg / other.deg
        else:
            raise NotImplementedError(f'Cannot divide distance by {type(other)}')

    def __mul__(self, other: float) -> 'Distance':
        if isinstance(other, (int, float)):
            return Distance(deg=self.deg * other)
        else:
            raise NotImplementedError(f'Cannot multiply Distance and {type(other)}')

    def __rmul__(self, other: float) -> 'Distance':
        return self.__mul__(other)

    def __eq__(self, other: 'Distance') -> bool:
        return abs(self.m - other.m) < 1e-9

    def __neq__(self, other: 'Distance') -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: 'Distance') -> bool:
        return self.m < other.m

    def __le__(self, other: 'Distance') -> bool:
        return self.m <= other.m

    def __gt__(self, other: 'Distance') -> bool:
        return self.m > other.m

    def __ge__(self, other: 'Distance') -> bool:
        return self.m >= other.m

    def __repr__(self) -> str:
        return f"{self.deg:.2f}°"


class Velocity:
    """A compact velocity unit."""

    def __init__(self,
                 *,
                 mps: float = 0.0,
                 meters_per_second: float = 0.0,
                 kmph: float = 0.0,
                 kilometers_per_hour: float = 0.0):
        # Note: distance is stored as meters per second.
        self._velocity = (
                mps + meters_per_second + (kmph + kilometers_per_hour) * 1000 / 3600)

    @property
    def mps(self) -> float:
        """Gets velocity in meters per second."""
        return self._velocity

    @property
    def meters_per_second(self) -> float:
        """Gets velocity in meters per second."""
        return self.mps

    @property
    def kmph(self) -> float:
        """Gets velocity in kilometers per hour."""
        return self._velocity * 3600 / 1000

    @property
    def kilometers_per_hour(self) -> float:
        """Gets velocity in kilometers per hour."""
        return self.kmph

    def __add__(self, other: 'Velocity') -> 'Velocity':
        if isinstance(other, Velocity):
            return Velocity(mps=self.mps + other.mps)
        else:
            raise NotImplementedError(f'Cannot add Velocity and {type(other)}')

    def __sub__(self, other: 'Velocity') -> 'Velocity':
        if isinstance(other, Velocity):
            return Velocity(mps=self.mps - other.mps)
        else:
            raise NotImplementedError(f'Cannot subtract Velocity and {type(other)}')

    def __mul__(self, other: datetime.timedelta) -> Distance:
        if isinstance(other, datetime.timedelta):
            # distance = velocity * time (for constant velocity).
            return Distance(m=self.mps * other.total_seconds())
        else:
            raise NotImplementedError(f'Cannot multiply velocity with {type(other)}')

    def __rmul__(self, other: datetime.timedelta) -> Distance:
        return self.__mul__(other)

    def __eq__(self, other: 'Velocity') -> bool:
        if isinstance(other, Velocity):
            # Note: we consider very similar velocities to be equal.
            return abs(self.mps - other.mps) < 1e-9
        else:
            raise ValueError(f'Cannot compare velocity and {type(other)}')

    def __repr__(self) -> str:
        return f'{self.mps} m/s'


class Energy(object):
    """A compact energy class."""

    def __init__(self, *,
                 watt_hours: float = 0.0,
                 joule: float = 0.0):
        self._joule = joule + watt_hours * 3600

    @property
    def watt_hours(self) -> float:
        return self._joule / 3600

    @property
    def joule(self) -> float:
        return self._joule

    def __add__(self, other: 'Energy') -> 'Energy':
        if isinstance(other, Energy):
            return Energy(joule=self.joule + other.joule)
        else:
            raise NotImplementedError(f'Cannot add Energy and {type(other)}')

    def __sub__(self, other: 'Energy') -> 'Energy':
        if isinstance(other, Energy):
            return Energy(joule=self.joule - other.joule)
        else:
            raise NotImplementedError(f'Cannot subtract Energy and {type(other)}')

    def __truediv__(self, other: 'Energy') -> float:
        if isinstance(other, Energy):
            return self.joule / other.joule
        else:
            raise NotImplementedError(f'Cannot divide Energy and {type(other)}')

    def __mul__(self, other: float) -> 'Energy':
        if isinstance(other, (int, float)):
            return Energy(joule=self.joule * other)
        else:
            raise NotImplementedError(f'Cannot multiply Energy and {type(other)}')

    def __rmul__(self, other: float) -> 'Energy':
        return self.__mul__(other)

    def __gt__(self, other: 'Energy') -> bool:
        if isinstance(other, Energy):
            return self.joule > other.joule
        else:
            raise ValueError(f'Cannot compare Energy and {type(other)}')

    def __eq__(self, other: 'Energy') -> bool:
        if isinstance(other, Energy):
            return self.joule == other.joule
        else:
            raise ValueError(f'Cannot compare Energy and {type(other)}')

    def __ge__(self, other: 'Energy') -> bool:
        if isinstance(other, Energy):
            return self.joule >= other.joule
        else:
            raise ValueError(f'Cannot compare Energy and {type(other)}')


class Power(object):
    """A compact power class."""

    def __init__(self, *, watts: float = 0.0):
        self._w = watts

    @property
    def watts(self) -> float:
        return self._w

    def __add__(self, other: 'Power') -> 'Power':
        if isinstance(other, Power):
            return Power(watts=self.watts + other.watts)
        else:
            raise NotImplementedError(f'Cannot add Power and {type(other)}')

    def __sub__(self, other: 'Power') -> 'Power':
        if isinstance(other, Power):
            return Power(watts=self.watts - other.watts)
        else:
            raise NotImplementedError(f'Cannot subtract Power and {type(other)}')

    def __mul__(self, other: datetime.timedelta) -> Energy:
        if isinstance(other, datetime.timedelta):
            return Energy(watt_hours=self.watts * timedelta_to_hours(other))
        else:
            raise NotImplementedError(f'Cannot multiply Power with {type(other)}')

    def __rmul__(self, other: datetime.timedelta) -> Energy:
        return self.__mul__(other)

    def __gt__(self, other: 'Power') -> bool:
        if isinstance(other, Power):
            return self.watts > other.watts
        else:
            raise ValueError(f'Cannot compare Power and {type(other)}')

    def __eq__(self, other: 'Power') -> bool:
        if isinstance(other, Power):
            return self.watts == other.watts
        else:
            raise ValueError(f'Cannot compare Power and {type(other)}')


class Mass(object):
    """A compact mass class."""

    def __init__(self, *, kilograms: float = 0.0, kg: float = 0.0):
        self._kg = kilograms + kg

    @property
    def kilograms(self) -> float:
        return self._kg

    @property
    def kg(self) -> float:
        return self._kg

    def __add__(self, other: 'Mass') -> 'Mass':
        if isinstance(other, Mass):
            return Mass(kg=self.kg + other.kg)
        else:
            raise NotImplementedError(f'Cannot add Mass and {type(other)}')

    def __sub__(self, other: 'Mass') -> 'Mass':
        if isinstance(other, Mass):
            return Mass(kg=self.kg - other.kg)
        else:
            raise NotImplementedError(f'Cannot subtract Mass and {type(other)}')

    @typing.overload
    def __truediv__(self, other: float) -> 'Mass':
        ...

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Mass(kg=self.kg / other)
        else:
            raise NotImplementedError(f'Cannot divide Mass by {type(other)}')

    def __mul__(self, other: float) -> 'Mass':
        if isinstance(other, (int, float)):
            return Mass(kg=self.kg * other)
        else:
            raise NotImplementedError(f'Cannot multiply Mass and {type(other)}')

    def __rmul__(self, other: float) -> 'Mass':
        return self.__mul__(other)

    def __eq__(self, other: 'Mass') -> bool:
        if isinstance(other, Mass):
            return abs(self.kg - other.kg) < 1e-9
        else:
            raise ValueError(f'Cannot compare Mass and {type(other)}')

    def __neq__(self, other: 'Mass') -> bool:
        if isinstance(other, Mass):
            return not self.__eq__(other)
        else:
            raise ValueError(f'Cannot compare Mass and {type(other)}')

    def __lt__(self, other: 'Mass') -> bool:
        if isinstance(other, Mass):
            return self.kg < other.kg
        else:
            raise ValueError(f'Cannot compare Mass and {type(other)}')

    def __le__(self, other: 'Mass') -> bool:
        if isinstance(other, Mass):
            return self.kg <= other.kg
        else:
            raise ValueError(f'Cannot compare Mass and {type(other)}')

    def __gt__(self, other: 'Mass') -> bool:
        if isinstance(other, Mass):
            return self.kg > other.kg
        else:
            raise ValueError(f'Cannot compare Mass and {type(other)}')

    def __ge__(self, other: 'Mass') -> bool:
        if isinstance(other, Mass):
            return self.kg >= other.kg
        else:
            raise ValueError(f'Cannot compare Mass and {type(other)}')


def relative_distance(x: Distance, y: Distance) -> Distance:
    # Assumes x, y are relative to the target, so the distance is simply the norm.
    return Distance(m=np.linalg.norm([x.m, y.m], ord=2).item())


def seconds_to_hours(s: float) -> float:
    return s / 3600.0


def timedelta_to_hours(d: datetime.timedelta) -> float:
    return seconds_to_hours(d.total_seconds())


def datetime_from_timestamp(timestamp: int) -> datetime.datetime:
    """Converts a given UTC timestamp into a datetime.

  The returned datetime includes timezone information.

  Args:
    timestamp: the timestamp (unix epoch; implicitly UTC).

  Returns:
    the corresponding datetime.
  """
    return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)


def get_posix_time_from_np64(np64_time_array: np.datetime64) -> np.array:
    """Helper function to transform """
    # transform from numpy datetime to POSIX time
    t_posix = (np64_time_array - np.datetime64(0, 's')) / np.timedelta64(1, 's')
    return t_posix


def get_datetime_from_np64(np64_time_array: np.datetime64) -> datetime.datetime:
    """Helper function to transform """
    # transform from numpy datetime to datetime
    t_posix = (np64_time_array - np.datetime64(0, 's')) / np.timedelta64(1, 's')
    dt_object = datetime.datetime.fromtimestamp(t_posix, datetime.timezone.utc)
    return dt_object


def posix_to_rel_seconds_in_year(posix_timestamp: float) -> float:
    """Helper function to map a posix_timestamp to it's relative seconds for the specific year (since 1st of January).
  This is needed because the interpolation function for the nutrients operates on relative timestamps as we take
  the average monthly nutrients for those as input.
  Args:
      posix_timestamp: a posix timestamp
  """
    # correction for extra long years because of Schaltjahre (makes it accurate 2020-2024, otherwise a couple of days off)
    correction_seconds = 13 * 24 * 3600
    # Calculate the relative time of the year in seconds
    return np.mod(posix_timestamp - correction_seconds, 365 * 24 * 3600)


from math import log10, floor
def round_to_sig_digits(x, sig_digit):
    if x == 0:
        return 0
    else:
        return round(x, sig_digit - int(floor(log10(abs(x)))) - 1)

import matplotlib
def format_datetime_x_axis(ax: plt.axis):
    """Helper function for better formatting the x_axis for datetimes."""
    locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = matplotlib.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
