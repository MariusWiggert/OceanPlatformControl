"""Provides the solar_rad function, which calcluates incoming radiation at a 
time, latitude, and longitude in the units W/m^2"""

import casadi as ca
import ocean_navigation_simulator.utils.solar_radiation.spa as spa


def solar_rad(t, lat, lon, pressure=1013.25, atmos_refract=0.5667, temp=12):
    """
    t: current time as ca.MX (and unixtime)

    lat: latitude as ca.MX
    lon: longitude as ca.MX

    pressure: pressure in millibars (default: one atmosphere or 1013.25)
    delta_t: not 100% sure but it's what is given by default

    """

    # this is always sea level since the system is always at sea :P
    elev = 0
    delta_t = 67.0
    numthreads = 0
    unixtime = t

    # We use the atmosphere corrected solar elevation angle.
    h = spa.radians(spa.solar_position_numpy(unixtime, lat, lon, elev, pressure, temp, delta_t,
                         atmos_refract, numthreads, sst=False, esd=False)[2])

    # We then take the solar angle and derive solar radiation from a basic sinusoidal model.
    # TODO: address admissibility concerns (see test.py)
    # We assume "middle-level" cloud cover
    a_i = 0.34
    b_i = 0.19

    # we need to bound the sine for cases where the solar elevation is in the nighttime, so we don't plug in 
    # zero into ca.log.
    bounded_sine = ca.fmax(ca.sin(h), 0.0001)

    return ca.fmax((1368.0 * bounded_sine) * (a_i + b_i * ca.log(bounded_sine)), 0)