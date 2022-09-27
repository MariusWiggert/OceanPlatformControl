import math


class InBounds:
    """Utils for checking validity of points"""

    def __init__(self, fieldset):
        self.fieldset = fieldset

    def valid_start_and_end(self, x_0, x_T):
        """Determines whether the given start (x_0) and target (x_T) are valid.

        For a start and end to be valid, they must be sufficiently far apart, and neither point can be in the ocean.

        Args:
            x_0:
                The starting point, a pair of longitude and latitude coordinates: [lon, lat].
            x_T:
                The target point, a pair of longitude and latitude coordinates: [lon, lat].
        Returns:
            A boolean.
        """
        if x_0 is None or x_T is None:
            return False
        return self.is_far_apart(x_0, x_T) and self.in_ocean(x_0) and self.in_ocean(x_T)

    def is_far_apart(self, x_0, x_T, sep=0.5):
        """Returns whether x_0 and x_T are sufficiently far apart

        Args:
            x_0:
                The starting point, a pair of longitude and latitude coordinates: [lon, lat].
            x_T:
                The target point, a pair of longitude and latitude coordinates: [lon, lat].
            sep:
                The minimum distance between the two points.
        Returns:
            A boolean.
        """
        lon, lat, lon_target, lat_target = x_0[0], x_0[1], x_T[0], x_T[1]
        dlon = lon_target - lon
        dlat = lat_target - lat
        mag = math.sqrt(dlon * dlon + dlat * dlat)  # mag is the distance between the two points.
        return mag > sep

    def in_ocean(self, point, offset=0.1):
        """Returns whether the point is in the ocean.

        Determines this by checking if the velocity is nonzero for this and ALL points that are "offset" distance
        about the point in the 8 directions.

        Args:
            point:
                A pair of longitude and latitude coordinates: [lon, lat].
            offset: A float which determines how far about the point to look. Increasing the value of offset will
                prevent points on the coast from being chosen.

        Returns:
            A boolean.
        """

        lon, lat = point[0], point[1]
        degrees = [0, offset, -offset]
        offsets = [(lon, lat) for lon in degrees for lat in degrees]
        for lon_offset, lat_offset in offsets:
            if self.zero_velocity(lon + lon_offset, lat + lat_offset):
                return False
        return True

    def out_of_bounds(self, coordinate, grid):
        """Determines whether the given coordinate (either lat or lon) is out of bounds for its respective grid.

        Returns:
            A boolean.
        """
        return coordinate < min(grid) or coordinate > max(grid)

    def zero_velocity(self, lon, lat):
        """Determines whether the (lon, lat) pair is zero velocity, i.e. on land.

        Returns:
            A boolean.
        """
        if self.out_of_bounds(lat, self.fieldset.U.grid.lat) or self.out_of_bounds(
            lon, self.fieldset.U.grid.lon
        ):
            return True
        x = self.fieldset.U.eval(0.0, 0.0, lat, lon)
        y = self.fieldset.V.eval(0.0, 0.0, lat, lon)
        return x == 0.0 and y == 0.0
