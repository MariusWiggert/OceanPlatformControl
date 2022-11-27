import dataclasses
import datetime
import math
from typing import List, Tuple, Union

import numpy as np

from ocean_navigation_simulator.utils import units


@dataclasses.dataclass
class SpatialPoint:
    """A dataclass containing variables that define the spatial position.

    Attributes:
      lon: The latitude position in degree
      lat: The longitude position in degree
    """

    lon: units.Distance
    lat: units.Distance

    def __post_init__(self):
        self._is_multi_agent = self.lon.is_array

    def distance(self, other) -> units.Distance:
        return units.Distance(
            deg=np.sqrt((self.lat.deg - other.lat.deg) ** 2 + (self.lon.deg - other.lon.deg) ** 2)
        )

    def haversine(self, other) -> units.Distance:
        """
        Calculate the great circle distance in degrees between two points
        on the earth (specified in decimal degrees)
        Taken from: https://stackoverflow.com/a/4913653
        """
        return units.Distance(
            rad=units.haversine_rad_from_deg(
                self.lon.deg, self.lat.deg, other.lon.deg, other.lat.deg
            )
        )

    def __array__(self):
        return np.array([self.lon.deg, self.lat.deg]).T  # states as columns if multiagent

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    def __repr__(self):
        if self._is_multi_agent:
            return f"[{np.array2string(self.lon.deg, formatter={'float': lambda x: f'{x:.5f}°'})}, \
                      {np.array2string(self.lat.deg, formatter={'float': lambda x: f'{x:.5f}°'})}]"
        else:
            return f"[{self.lon.deg:5f}°,{self.lat.deg:.5f}°]"


@dataclasses.dataclass
class SpatioTemporalPoint:
    # TODO: implement nice way to transform a list of those to numpy and back: https://kplauritzen.dk/2021/08/11/convert-dataclasss-np-array.html
    """A dataclass containing SpatioTemporalPoint variables..

    Attributes:
      lon: The latitude position in degree
      lat: The longitude position in degree
      date_time: The current time.
    """
    lon: units.Distance
    lat: units.Distance
    date_time: Union[datetime.datetime, np.ndarray]

    def __post_init__(self):
        self.vect_timestamp = np.vectorize(datetime.datetime.timestamp)
        self.vect_strftime = np.vectorize(datetime.datetime.strftime)
        self.rmv_tzinfo = np.vectorize(datetime.datetime.replace)
        self._is_multi_agent = self.lon.is_array

    def __array__(self):
        if self._is_multi_agent:
            return np.array(
                [self.lon.deg, self.lat.deg, self.vect_timestamp(self.date_time)]
            ).T  # states as columns
        else:
            return np.array([self.lon.deg, self.lat.deg, self.date_time.timestamp()])

    def __len__(self):
        if self._is_multi_agent:
            return self.__array__().shape[1]  # matrix where number of states are on the columns
        else:
            return self.__array__().shape[0]  # vector of lon,lat,date_time

    def __getitem__(self, item):
        if self._is_multi_agent:
            return self.__array__()[
                :, item
            ]  # extract the state corresponding to index item (for all platforms)
        else:
            return self.__array__()[item]

    def get_datetime_bounds(self) -> Tuple[datetime.datetime, datetime.datetime]:
        if self._is_multi_agent:
            return np.min(self.date_time), np.max(self.date_time)
        else:
            return self.date_time, self.date_time

    def date_time_to_datetime64(self):
        date_time_no_tz = self.rmv_tzinfo(self.date_time, tzinfo=None)
        return date_time_no_tz.astype("datetime64[s]")

    def distance(self, other) -> units.Distance:
        return self.to_spatial_point().distance(other)

    def haversine(self, other) -> units.Distance:
        return self.to_spatial_point().haversine(other)

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

    def to_spatio_temporal_casadi_input(self) -> List[float]:
        """Helper function to produce a list [posix_time, lat, lon] to feed into casadi."""
        if self._is_multi_agent:
            return np.array(self)[:, [2, 1, 0]].T  # [t, lat,lon] x [nb_platforms]
        else:
            return [self.date_time.timestamp(), self.lat.deg, self.lon.deg]

    def __repr__(self):
        if self._is_multi_agent:
            return f"[{np.array2string(self.lon.deg, formatter={'float': lambda x: f'{x:.5f}°'})}, \
                      {np.array2string(self.lat.deg, formatter={'float': lambda x: f'{x:.5f}°'})}], \
                      {np.array2string(self.vect_strftime(self.date_time, '%Y-%m-%d %H:%M:%S'))}]"
        else:
            return f"[{self.lon.deg:5f}°,{self.lat.deg:.5f}°,{self.date_time.strftime('%Y-%m-%d %H:%M:%S')}]"


@dataclasses.dataclass
class PlatformState:
    """A dataclass containing variables relevant to the platform state.

    Attributes:
      lon: The latitude position in degree
      lat: The longitude position in degree
      battery_charge: The amount of energy stored on the batteries.
      seaweed_mass: The amount of seaweed biomass on the system.
      date_time: The current time.
    """

    lon: units.Distance
    lat: units.Distance
    date_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    battery_charge: units.Energy = units.Energy(joule=100)
    seaweed_mass: units.Mass = units.Mass(kg=100)

    def __array__(self):
        return np.array(
            [
                self.lon.deg,
                self.lat.deg,
                self.date_time.timestamp(),
                self.battery_charge.joule,
                self.seaweed_mass.kg,
            ]
        )

    def __len__(self):
        return self.__array__().shape[0]

    def __getitem__(self, item):
        return self.__array__()[item]

    @staticmethod
    def from_numpy(numpy_array):
        """Helper function to initialize a PlatformState based on numpy arraay.
        Args:
            numpy_array
        Returns:
            PlatformAction object
        """
        return PlatformState(
            lon=units.Distance(deg=numpy_array[0]),
            lat=units.Distance(deg=numpy_array[1]),
            date_time=datetime.datetime.fromtimestamp(numpy_array[2], tz=datetime.timezone.utc),
            battery_charge=units.Energy(joule=numpy_array[3]),
            seaweed_mass=units.Mass(kg=numpy_array[4]),
        )

    @staticmethod
    def from_spatio_temporal_point(point: SpatioTemporalPoint):
        return PlatformState(lon=point.lon, lat=point.lat, date_time=point.date_time)

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

    def to_spatio_temporal_point(self) -> SpatioTemporalPoint:
        """Helper function to just extract the spatio-temporal point."""
        return SpatioTemporalPoint(lon=self.lon, lat=self.lat, date_time=self.date_time)

    def to_spatio_temporal_casadi_input(self) -> List[float]:
        """Helper function to produce a list [posix_time, lat, lon] to feed into casadi."""
        return [self.date_time.timestamp(), self.lat.deg, self.lon.deg]

    def __repr__(self):
        return "Platform State[lon: {x} deg, lat: {y} deg, date_time: {t}, battery_charge: {b} Joule, seaweed_mass: {m} kg]".format(
            x=self.lon.deg,
            y=self.lat.deg,
            b=self.battery_charge.joule,
            m=self.seaweed_mass.kg,
            t=self.date_time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def distance(self, other) -> float:
        return self.to_spatial_point().distance(other)


@dataclasses.dataclass
class PlatformStateSet:
    platform_list: List[PlatformState]

    def __array__(self):
        return np.array(self.platform_list)  # rows are the number of platforms

    def __len__(self):
        return len(self.platform_list)

    def __getitem__(self, platform_id):
        #     return np.array(self.states[platform_id])
        return self.platform_list[platform_id]

    def __post_init__(self):
        self.lon = units.Distance(deg=np.array(self.platform_list)[:, 0])
        self.lat = units.Distance(deg=np.array(self.platform_list)[:, 1])
        # datetime does not support array, whilst np.datetime64 has no timestamps....
        self.date_time = np.array([platform.date_time for platform in self.platform_list])
        self.battery_charge = units.Energy(joule=np.array(self.platform_list)[:, 3])
        self.seaweed_masse = units.Mass(kg=np.array(self.platform_list)[:, 4])

    def to_spatial_point(self) -> SpatialPoint:
        """Helper function to just extract the spatial point."""
        return SpatialPoint(lon=self.lon, lat=self.lat)

    def to_spatio_temporal_point(self) -> SpatioTemporalPoint:
        """Helper function to just extract the spatial point."""
        return SpatioTemporalPoint(lon=self.lon, lat=self.lat, date_time=self.date_time)

    def get_nodes_list(self):
        return list(range(len(self.platform_list)))

    def get_distance_btw_platforms(self, from_nodes, to_nodes):
        lon_from, lat_from = self.lon[from_nodes], self.lat[from_nodes]
        lon_to, lat_to = self.lon[to_nodes], self.lat[to_nodes]
        from_spatial_point = SpatialPoint(lon=lon_from, lat=lat_from)
        return from_spatial_point.distance(SpatialPoint(lon=lon_to, lat=lat_to))

    @staticmethod
    def from_numpy(np_array):
        platform_list = [PlatformState.from_numpy(np_array[k, :]) for k in range(np_array.shape[0])]
        return PlatformStateSet(platform_list=platform_list)
