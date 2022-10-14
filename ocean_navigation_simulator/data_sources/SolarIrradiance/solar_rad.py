"""Provides the solar_rad function, which calcluates incoming radiation at a
time, latitude, and longitude in the units W/m^2"""
import casadi as ca
import numpy as np

import ocean_navigation_simulator.data_sources.SolarIrradiance.spa as spa
import ocean_navigation_simulator.data_sources.SolarIrradiance.spa_casadi as spa_casadi


def solar_rad_ca(t_lat_lon, pressure=1013.25, atmos_refract=0.5667, temp=12):
    """Casadi all through version
    t_lat_lon array

    pressure: pressure in millibars (default: one atmosphere or 1013.25)
    delta_t: not 100% sure but it's what is given by default

    """

    # this is always sea level since the system is always at sea :P
    # We use the atmosphere corrected solar elevation angle.
    h = spa_casadi.radians(
        spa_casadi.solar_position_numpy(
            t_lat_lon[0],
            t_lat_lon[1],
            t_lat_lon[2],
            elev=0,
            pressure=pressure,
            temp=temp,
            delta_t=67.0,
            atmos_refract=atmos_refract,
            numthreads=0,
            sst=False,
            esd=False,
        )[2]
    )

    # We then take the solar angle and derive solar radiation from a basic sinusoidal model.
    # TODO: address admissibility concerns (see test.py)
    # We assume "middle-level" cloud cover
    a_i = 0.34
    b_i = 0.19

    # we need to bound the sine for cases where the solar elevation is in the nighttime, so we don't plug in
    # zero into np.log.
    bounded_sine = ca.fmax(ca.sin(h), 0.0001)

    return ca.fmax((1368.0 * bounded_sine) * (a_i + b_i * ca.log(bounded_sine)), 0)


def solar_rad(unixtime, lat, lon, pressure=1013.25, atmos_refract=0.5667, temp=12):
    """
    t: current time as in POSIX numpy array

    lat: latitude as numpy array
    lon: longitude as numpy array

    pressure: pressure in millibars (default: one atmosphere or 1013.25)
    delta_t: not 100% sure but it's what is given by default

    """

    # this is always sea level since the system is always at sea :P
    # We use the atmosphere corrected solar elevation angle.
    h = spa.radians(
        spa.solar_position_numpy(
            unixtime,
            lat,
            lon,
            elev=0,
            pressure=pressure,
            temp=temp,
            delta_t=67.0,
            atmos_refract=atmos_refract,
            numthreads=0,
            sst=False,
            esd=False,
        )[2]
    )

    # We then take the solar angle and derive solar radiation from a basic sinusoidal model.
    # TODO: address admissibility concerns (see test.py)
    # We assume "middle-level" cloud cover
    a_i = 0.34
    b_i = 0.19

    # we need to bound the sine for cases where the solar elevation is in the nighttime, so we don't plug in
    # zero into np.log.
    bounded_sine = np.maximum(np.sin(h), 0.0001)

    return np.maximum((1368.0 * bounded_sine) * (a_i + b_i * np.log(bounded_sine)), 0)
