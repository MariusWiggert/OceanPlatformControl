import matplotlib.pyplot as plt
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, Field, Variable
from parcels.plotting import plotfield
from parcels import rng as random
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime,timedelta
import os, math
import netCDF4
import matplotlib.pylab as plt
import pickle


def get_hycom_fieldset(ncfile):
    filenames = {'U': ncfile,
                 'V': ncfile}

    variables = {'U': 'water_u',
                 'V': 'water_v'}

    dimensions = {'lat': 'lat',
                  'lon': 'lon',
                  'time': 'time',
                  'depth': 'depth'}
    # Note deferred_load=True only loads the time steps immediately before & after the current time. This is to save memory in the RAM
    # at some point we might use that again/build our own
    return FieldSet.from_netcdf(filenames, variables, dimensions, mesh='spherical', allow_time_extrapolation=True,
                                deferred_load=True) # indices={'U':})


def make_movie_fieldset(fieldset, particle_ncfile, domain, prefix):
    import subprocess

    plot_path = os.path.join(prefix, 'plot')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    ncp = netCDF4.Dataset(particle_ncfile)
    dt = ncp.variables['time'][:].data
    nframes = len(dt)
    for itime, dtime in enumerate(dt):
        print('Writing frame %i of %i...' % (itime, nframes))
        fig, ax = plot_field_w_particles(ncp, fieldset, 'vector', domain, itime, dtime)
        plt.savefig(os.path.join(plot_path, '%05i_mframe.png' % itime), dpi=400, bbox='tight')
        plt.close(fig)
        plt.clf()
    out_mp4 = os.path.join(prefix, prefix + '.mp4')
    subprocess.call(
        "/usr/local/bin/ffmpeg -r 8 -threads 8 -i %s/%%05d_mframe.png -s 1280x1060 -vcodec libx264 -pix_fmt yuv420p -y %s" % (
        plot_path, out_mp4),
        shell=True)


def plot_field_w_particles(ncp, fieldset, field, domain, itime, dtime, size=10):
    if field == 'vector':
      field = fieldset.UV
    elif not isinstance(field, Field):
      field = getattr(fieldset, field)
    _, fig, ax, _ = plotfield(field=field, animation=False,
                                    show_time=dtime, domain=domain,
                                    projection=None, land=True,
                                    vmin=0, vmax=0.8, savefile=None,
                                    titlestr='Particles and ',
                                    depth_level=0,
                                    no_draw=True)

    plon = ncp.variables['lon'][itime].data
    plat = ncp.variables['lat'][itime].data
    ax.scatter(plon, plat, s=size, color='black')#, zorder=20)
    plt.xticks(rotation=45)
    return fig, ax

# def subset_nc_file(lower_left, upper_right, particle_ncfile):


def plot_domain_from_nc(particle_ncfile):
  ncp = netCDF4.Dataset(particle_ncfile)
  plon = ncp.variables['lon'][:].data
  plat = ncp.variables['lat'][:].data
  ncp.close()
  return plot_domain(plon, plat)


def plot_vec_fied(fieldset):
    X = fieldset.U.lon
    Y = fieldset.U.lat

    U = fieldset.U.data[0, 0, :, :]
    V = fieldset.V.data[0, 0, :, :]

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)

    plt.show()


def plot_domain(plon, plat):
  """
  return bounds in dictionary form of particle spread
  """
  lon_max = np.nanmax(plon)
  lon_min = np.nanmin(plon)
  lat_max = np.nanmax(plat)
  lat_min = np.nanmin(plat)
  return {'N': lat_max, 'S': lat_min, 'W': lon_min, 'E': lon_max}
  # lon5p = np.max([1.0,(lon_max-lon_min)*0.05]) # min one degree
  # lat5p = np.max([1.0,(lat_max-lat_min)*0.05]) # min one degree
  # return {'N':lat_max+lat5p,'S':lat_min-lat5p,'W':lon_min-lon5p,'E':lon_max+lon5p}