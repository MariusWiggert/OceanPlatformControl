import numpy as np

# def get_solarposition(time, latitude, longitude,
#                      altitude=None, pressure=None,
#                      method='nrel_numpy',
#                      temperature=12, **kwargs):

# we assume altitude = 0m (sea level; when you're on the ocean, it's a bit hard _not_ to be at sea level)
# pressure = 101325 Pa
# temperature = 12 deg C

# e0 = topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
# delta_e = atmosphereic_refraction_correction(pressure, temp, e0, atmos_refract)


def get_solarposition(time, lat, lon,
                      altitude=0, pressure=101325/100,
                      temperature=12, **kwargs):
    atmos_refract = None
    atmos_refract = atmos_refract or 0.5667
    elev = altitude
    #ephem_df = spa_python(time, lat, lon, altitude, pressure, temperature, how='numpy', **kwargs)
    #return ephem_df
    unixtime = time
    delta_t = 67.0
    numthreads = -1
    app_zenith, zenith, app_elevation, elevation, azimuth, eot = \
        solar_position_numpy(unixtime, lat, lon, elev, pressure, temperature,
                           delta_t, atmos_refract, numthreads)


