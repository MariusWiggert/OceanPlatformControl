import casadi as ca
from datetime import datetime, timedelta, timezone
# import netCDF4
# import h5netcdf.legacyapi as netCDF4
import xarray as xr
import numpy as np
import math
import os
import warnings
from scipy import interpolate


def copernicusmarine_datastore(dataset, username, password):
    """Helper Function to establish an opendap session with copernicus data."""
    from pydap.client import open_url
    from pydap.cas.get_cookies import setup_session
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session)) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session)) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    return data_store


def convert_to_lat_lon_time_bounds(x_0, x_T, deg_around_x0_xT_box,
                                   temp_horizon_in_h=120, hours_to_hj_solve_timescale=3600):
    """
    We want to switch over to using lat-lon bounds but a lot of functions in the code already use x_0 and x_T with the
    deg_around_x0_xT_box, so for now use this to convert.
    Args:
        x_0: [lat, lon, charge, timestamp]
        x_T: [lon, lat] goal locations
        deg_around_x0_xT_box: buffer around the box in degrees
        temp_horizon_in_h: maximum temp_horizon to look ahead of x_0 time in hours
        hours_to_hj_solve_timescale: factor to multiply hours with to get to current field temporal units.

    Returns:
        t_interval: if time-varying: [t_0, t_T] as utc datetime objects
                                     where t_0 and t_T are the start and end respectively
        lat_bnds: [y_lower, y_upper] in degrees
        lon_bnds: [x_lower, x_upper] in degrees
    """
    if temp_horizon_in_h is None:
        t_interval = [datetime.fromtimestamp(x_0[3], timezone.utc), None]
    else:
        t_interval = [datetime.fromtimestamp(x_0[3], timezone.utc),
                      datetime.fromtimestamp(x_0[3] + temp_horizon_in_h * hours_to_hj_solve_timescale, timezone.utc)]
    lat_bnds = [min(x_0[1], x_T[1]) - deg_around_x0_xT_box, max(x_0[1], x_T[1]) + deg_around_x0_xT_box]
    lon_bnds = [min(x_0[0], x_T[0]) - deg_around_x0_xT_box, max(x_0[0], x_T[0]) + deg_around_x0_xT_box]

    return t_interval, lat_bnds, lon_bnds


def get_interpolation_func(grids_dict, u_data, v_data, type='bspline', fixed_time_index=None):
    """ Creates 2D or 3D cassadi interpolation functions to use in the simulator."""
    # fit interpolation function
    if fixed_time_index is None:    # time varying current
        # U field varying
        u_curr_func = ca.interpolant('u_curr', type,
                                     [grids_dict['t_grid'], grids_dict['y_grid'], grids_dict['x_grid']],
                                     u_data.ravel(order='F'))
        # V field varying
        v_curr_func = ca.interpolant('v_curr', type,
                                     [grids_dict['t_grid'], grids_dict['y_grid'], grids_dict['x_grid']],
                                     v_data.ravel(order='F'))
    else:      # fixed current
        # U field fixed
        u_curr_func = ca.interpolant('u_curr', type, [grids_dict['y_grid'], grids_dict['x_grid']], u_data.ravel(order='F'))
        # V field fixed
        v_curr_func = ca.interpolant('v_curr', type, [grids_dict['y_grid'], grids_dict['x_grid']], v_data.ravel(order='F'))

    return u_curr_func, v_curr_func


# define overarching loading and sub-setting function
def get_current_data_subset(t_interval, lat_interval, lon_interval, data_source, max_temp_in_h=240):
    """ Function to get a subset of current data from the files referenced in the files_dict.
    Inputs:
        t_interval              if time-varying: [t_0, t_T] in POSIX time
                                where t_0 and t_T are the start and end timestamps respectively
                                if t_T is None and we use a file, the full available time is returned.
        lat_interval            [y_lower, y_upper] in degrees
        lon_interval            [x_lower, x_upper] in degrees

        data_source             A dict object, always containing "data_source_type" and "content".
                                See problem.init() for description.

        max_temp_in_h           As multiple daily files hindcast data can be very large, limit the time horizon if t_interval[1] is None.

    Outputs:
        grids_dict              dict containing x_grid, y_grid, t_grid, and spatial_land_mask (2D X,Y array)
        u_data                  [T, Y, X] matrix of the ocean currents in x direction in m/s
        v_data                  [T, Y, X] matrix of the ocean currents in y direction in m/s
    """

    # Step 1: check in which data_source_type case we are and call the respective function
    if data_source['data_source_type'] == 'cop_opendap':
        grids_dict, u_data, v_data = get_current_data_subset_from_cop_opendap(t_interval, lat_interval, lon_interval,
                                                                              data_source)
    elif data_source['data_source_type'] == 'multiple_daily_nc_files':
        grids_dict, u_data, v_data = get_current_data_subset_from_daily_files(t_interval, lat_interval, lon_interval,
                                                                              data_source['content'], max_temp_in_h=max_temp_in_h)
    elif data_source['data_source_type'] == 'single_nc_file':
        # check if forecast_idx
        if 'current_forecast_idx' in data_source:
            file_idx = data_source['current_forecast_idx']
        else:
            file_idx = 0
        grids_dict, u_data, v_data = get_current_data_subset_from_single_file(t_interval, lat_interval, lon_interval,
                                                                              file_dict=data_source['content'][file_idx])
    elif data_source['data_source_type'] == 'analytical_function':
        grids_dict, u_data, v_data = data_source['content'].get_subset_from_analytical_field(t_interval, lat_interval, lon_interval)
        grids_dict['not_plot_land'] = True
    else:
        raise ValueError("file_dicts is neither multiple_daily_nc_files-file, nor cop_opendap, nor single_nc_file."
                         " Check source files.")

    # Step 2: log what data has been subsetted
    if data_source['data_source_type'] != 'analytical_function':
        print("Subsetted data from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
            start=datetime.utcfromtimestamp(grids_dict['t_grid'][0]).strftime('%Y-%m-%d %H:%M:%S UTC'),
            end=datetime.utcfromtimestamp(grids_dict['t_grid'][-1]).strftime('%Y-%m-%d %H:%M:%S UTC'),
            n_steps=len(grids_dict['t_grid']), time=(grids_dict['t_grid'][1] - grids_dict['t_grid'][0]) / 3600.))
    else:
        print("Sub-setted data from analytical_function from {start} to {end} in {n_steps} time steps".format(
            start=grids_dict['t_grid'][0],
            end=grids_dict['t_grid'][-1],
            n_steps=len(grids_dict['t_grid'])))

    return grids_dict, u_data, v_data


# Helper functions for the general subset function
def get_current_data_subset_from_cop_opendap(t_interval, lat_interval, lon_interval, data_source):
    """Helper function to get """
    # modify t_interval to be timezone naive because that's what the opendap subsetting takes
    t_interval_naive = [time.replace(tzinfo=None) for time in t_interval]
    t_interval_naive[0] = t_interval_naive[0] - timedelta(hours=1)
    t_interval_naive[1] = t_interval_naive[1] + timedelta(hours=1)
    subsetted_frame = data_source['content'].sel(time=slice(t_interval_naive[0], t_interval_naive[1]),
                                                 latitude=slice(lat_interval[0], lat_interval[1]),
                                                 longitude=slice(lon_interval[0], lon_interval[1]))

    # grids_dict containing x_grid, y_grid, t_grid (in POSIX), and spatial_land_mask (2D array)
    u_data_masked = np.ma.masked_invalid(subsetted_frame['uo'].data)
    v_data_masked = np.ma.masked_invalid(subsetted_frame['vo'].data)
    grids_dict = {'x_grid': subsetted_frame.longitude.data, 'y_grid': subsetted_frame.latitude.data,
                  't_grid': (subsetted_frame.time.data - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s'),
                  'spatial_land_mask': u_data_masked[0, :, :].mask}

    u_data = u_data_masked.filled(fill_value=0.)
    v_data = v_data_masked.filled(fill_value=0.)
    return grids_dict, u_data, v_data


def get_current_data_subset_from_single_file(t_interval, lat_interval, lon_interval, file_dict):
    """Subsetting data from a single file."""
    # Step 1: Open nc file
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    f = xr.open_dataset(file_dict['file'])

    # Step 2: Extract the grids
    x_grid = f.variables['lon'].data
    y_grid = f.variables['lat'].data
    t_grid = get_abs_time_grid_from_hycom_file(f)

    # Step 3: get the subsetting indices for space and time
    ygrid_inds = np.where((lat_interval[0] <= y_grid) & (y_grid <= lat_interval[1]))[0]
    ygrid_inds = add_element_front_and_back_if_possible(y_grid, ygrid_inds)
    xgrid_inds = np.where((lon_interval[0] <= x_grid) & (x_grid <= lon_interval[1]))[0]
    xgrid_inds = add_element_front_and_back_if_possible(x_grid, xgrid_inds)
    if t_interval[1] is None:  # take full file time contained in the file from t_interval[0] onwards
        tgrid_inds = np.where((t_grid >= t_interval[0].timestamp()))[0]
    else:  # subset ending time
        tgrid_inds = np.where((t_grid >= t_interval[0].timestamp()) & (t_grid <= t_interval[1].timestamp()))[0]
    tgrid_inds = add_element_front_and_back_if_possible(t_grid, tgrid_inds)

    # Step 3.1: create grid and use that as sanity check if any relevant data is contained in the file
    grids_dict = {'x_grid': x_grid[xgrid_inds], 'y_grid': y_grid[ygrid_inds], 't_grid': t_grid[tgrid_inds]}

    # Step 3.2: Advanced sanity check if only partial area is contained in file
    grids_interval_sanity_check(grids_dict, lat_interval, lon_interval, t_interval)

    # Step 4: extract data
    # Note: HYCOM is [tdim, zdim, ydim, xdim]
    if len(f.variables['water_u'].shape) == 4:  # if there is a depth dimension in the dataset
        u_data = f.variables['water_u'].data[tgrid_inds, 0, ...][:, ygrid_inds, ...][..., xgrid_inds]
        v_data = f.variables['water_v'].data[tgrid_inds, 0, ...][:, ygrid_inds, ...][..., xgrid_inds]
        # adds a mask where there's a nan (on-land)
        u_data = np.ma.masked_invalid(u_data)
        v_data = np.ma.masked_invalid(v_data)
    else:
        raise ValueError("Current data in nc file does not have 4 dimensions. Check file.")

    # Step 5: add the spatial land mask
    grids_dict['spatial_land_mask'] = u_data[0, :, :].mask

    # close netCDF file
    f.close()

    return grids_dict, u_data.filled(fill_value=0.), v_data.filled(fill_value=0.)


def get_current_data_subset_from_daily_files(t_interval, lat_interval, lon_interval, file_dicts, max_temp_in_h=120):
    """Sub-setting data from a list of daily file_dicts."""

    # Step 0.1: if None is put in, subset max_temp_in_h into the future
    if t_interval[1] is None:
        t_interval[1] = t_interval[0] + timedelta(hours=max_temp_in_h)

    # Step 0.2 round up t_interval[1] to the full hour, otherwise there can be issues with the logic below.
    if t_interval[1].minute > 0 or t_interval[1].second > 0 or t_interval[1].microsecond > 0:
        t_interval[1] = t_interval[1].replace(microsecond=0, second=0, minute=0) + timedelta(hours=1)

    # Step 1: filter all dicts that are needed for this time interval
    filter_func = lambda dic: not (
            # is the end time of the dict smaller than start of the time interval - 1 hour (because 1h buffer)
            dic['t_range'][1] < t_interval[0] - timedelta(hours=1) \
            # or is the start of the file bigger than the ending time + 1 hour (to have 1h buffer)
            or dic['t_range'][0] > t_interval[1] + timedelta(hours=1))
    time_interval_dicts = list(filter(filter_func, file_dicts))
    # Basic sanity check
    if len(time_interval_dicts) == 0:
        raise ValueError("No files found in the file_dicts for the requested t_interval.")

    # Step 2: Prepare the stacking loop by getting the x, y grids and subsetting indices in x, y
    # Note: these stay constant across files in this case where all files have same lat-lon range

    # Step 2.1: open the first file and get the x and y grid
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    f = xr.open_dataset(time_interval_dicts[0]['file'])
    x_grid = f.variables['lon'].data
    y_grid = f.variables['lat'].data

    # close netCDF file
    f.close()

    # Step 2.2: get the respective indices of the lat, lon subset from the file grids
    ygrid_inds = np.where((y_grid >= lat_interval[0]) & (y_grid <= lat_interval[1]))[0]
    ygrid_inds = add_element_front_and_back_if_possible(y_grid, ygrid_inds)
    xgrid_inds = np.where((x_grid >= lon_interval[0]) & (x_grid <= lon_interval[1]))[0]
    xgrid_inds = add_element_front_and_back_if_possible(x_grid, xgrid_inds)

    # Step 2.3 initialze t_grid stacking variable
    full_t_grid = []

    # Step 3: iterate over all files in order and stack the current data and absolute t_grids
    for idx in range(len(time_interval_dicts)):
        # Step 3.0: load the current data file
        f = xr.open_dataset(time_interval_dicts[idx]['file'])
        # set the default start and end time
        start_hr, end_hr = 0, 24

        # Step 3.1: do the time-subsetting
        # Case 1: file is first -- get data from the file from the hour before or at t_0
        if idx == 0:
            start_hr = math.floor(
                (t_interval[0] - time_interval_dicts[idx]['t_range'][0]).seconds/ 3600)
        # Case 2: file is last -- get data from file until or after the hour t_T
        if idx == len(time_interval_dicts) - 1:
            end_hr = math.ceil(
                (t_interval[1] - time_interval_dicts[idx]['t_range'][0]).seconds / 3600) + 1

        # Step 3.2: extract data from the file
        u_data_raw = f.variables['water_u'].data[start_hr:end_hr, 0, ygrid_inds, ...][..., xgrid_inds]
        v_data_raw = f.variables['water_v'].data[start_hr:end_hr, 0, ygrid_inds, ...][..., xgrid_inds]
        u_data = np.ma.masked_where(np.isnan(u_data_raw), u_data_raw)
        v_data = np.ma.masked_where(np.isnan(v_data_raw), v_data_raw)

        # Step 3.3: stack the sub-setted abs_t_grid and current data
        full_t_grid = full_t_grid + [time_interval_dicts[idx]['t_range'][0].timestamp() + i * 3600 for i in
                                     range(start_hr, end_hr)]

        if idx == 0:
            full_u_data = u_data.filled(fill_value=0.)
            full_v_data = v_data.filled(fill_value=0.)
        else:
            full_u_data = np.concatenate((full_u_data, u_data.filled(fill_value=0.)), axis=0)
            full_v_data = np.concatenate((full_v_data, v_data.filled(fill_value=0.)), axis=0)

        # close netCDF file
        f.close()

    # Step 4: from the last sub-setting of u_data -> get the land mask!
    land_mask = np.ma.masked_invalid(u_data[0,:,:]).mask

    # Step 5: create dict to output
    grids_dict = {'x_grid': x_grid[xgrid_inds], 'y_grid': y_grid[ygrid_inds], 't_grid': np.array(full_t_grid),
                  'spatial_land_mask': land_mask}

    # Step 6: Advanced sanity check if only partial area is contained in file
    grids_interval_sanity_check(grids_dict, lat_interval, lon_interval, t_interval)

    # Step 7: check if sizes of the data and grids work out, otherwise uninterpretable error by casadi
    if full_u_data.shape[0] != len(grids_dict['t_grid']) \
            or full_u_data.shape[1] != len(grids_dict['y_grid']) \
            or full_u_data.shape[2] != len(grids_dict['x_grid']):
        raise ValueError("Some bug in the get_current_data_subset_from_daily_files function."
                         " Grid_dict to data sizes don't work out.")

    # Step 8: return the grids_dict and the stacked data
    # Note: we fill in the on-land values with 0 which is ok near shore (linear interpolation to 0).
    #       and the simulator will recognize and get stuck if we're inside.
    return grids_dict, full_u_data, full_v_data


# Functions to do interpolation of the current data
# general interpolation function
def spatio_temporal_interpolation(grids_dict, u_data, v_data,
                                  temp_res_in_h=None, spatial_shape=None, spatial_kind='linear'):
    """Spatio-temporal interpolation of the current data to a new temp_res_in_h and new spatial_shape.
    Inputs:
    - grids_dict            containing at least x_grid', 'y_grid'
    - u_data, v_data        [T, Y, X] matrix of the current data
    - temp_res_in_h         desired temporal resolution of the data
    - spatial_shape         Desired spatial shape as tuple or list e.g. (<# of y_points>, <# of x_points>)
    - spatial_kind          which interpolation to use, options are 'cubic', 'linear'

    Outputs:
    - grids_dict                updated grids_dict
    - u_data_new, v_data_new    defined just like input
    """
    # copy dict object to not inadvertantly change the original
    new_grids_dict = grids_dict.copy()
    if temp_res_in_h is not None:
        new_grids_dict['t_grid'], u_data, v_data = temporal_interpolation(grids_dict, u_data, v_data, temp_res_in_h)
    if spatial_shape is not None:
        new_grids_dict['x_grid'], new_grids_dict['y_grid'], u_data, v_data = spatial_interpolation(grids_dict, u_data, v_data,
                                                             target_shape=spatial_shape, kind=spatial_kind)
    return new_grids_dict, u_data, v_data

# spatial interpolation function
def spatial_interpolation(grid_dict, u_data, v_data, target_shape, kind='linear'):
    """Doing spatial interpolation to a specific spatial shape e.g. (100, 100).
    Inputs:
    - grid_dict             containing at least x_grid', 'y_grid'
    - u_data, v_data        [T, Y, X] matrix of the current data
    - target_shape          Shape as tuple or list e.g. (<# of y_points>, <# of x_points>)
    - kind                  which interpolation to use, options are 'cubic', 'linear'

    Outputs:
    - x_grid_new, x_grid_new    arrays of the new x and y grid
    - u_data_new, v_data_new    defined just like input
    """
    # Step 1: create the new x and y axis vectors
    x_grid_new = np.arange(target_shape[1]) * (grid_dict['x_grid'][-1] - grid_dict['x_grid'][0]) / (
            target_shape[1] - 1) + grid_dict['x_grid'][0]
    y_grid_new = np.arange(target_shape[0]) * (grid_dict['y_grid'][-1] - grid_dict['y_grid'][0]) / (
            target_shape[0] - 1) + grid_dict['y_grid'][0]

    # create the arrays to fill in with the new-resolution data
    u_data_new = np.zeros(shape=(u_data.shape[0], target_shape[0], target_shape[1]))
    v_data_new = np.zeros(shape=(u_data.shape[0], target_shape[0], target_shape[1]))

    # Step 2: iterate over the time axis to create the new u and v data
    for t_idx in range(u_data.shape[0]):
        # run spatial interpolation in 2D along the new axis
        u_data_new[t_idx, :, :] = interpolate.interp2d(grid_dict['x_grid'], grid_dict['y_grid'],
                                                       u_data[t_idx, :, :], kind=kind)(x_grid_new, y_grid_new)
        v_data_new[t_idx, :, :] = interpolate.interp2d(grid_dict['x_grid'], grid_dict['y_grid'],
                                                       v_data[t_idx, :, :], kind=kind)(x_grid_new, y_grid_new)

    return x_grid_new, y_grid_new, u_data_new, v_data_new

# temporal interpolation function
def temporal_interpolation(grids_dict, u_data, v_data, temp_res_in_h):
    """Doing linear temporal interpolation of the u and v data for a specific resolution.
       Inputs:
       - grid_dict             containing at least x_grid', 'y_grid'
       - u_data, v_data        [T, Y, X] matrix of the current data
       - temp_res_in_h         desired temporal resolution in hours

       Outputs:
       - t_grid_new                arrays of the new t grid
       - u_data_new, v_data_new    defined just like input
       """
    # check
    if temp_res_in_h <= 0:
        raise ValueError("Temporal resolution must be positive")
    t_span_in_h = (grids_dict['t_grid'][-1] - grids_dict['t_grid'][0]) / 3600
    # TODO: Think if we want to implement temporal aggregation
    # # Case 1: aggregation, we need to average over the values
    # if temp_res_in_h >= (time_span_in_s/(3600*grids_dict['t_grid'].shape[0])):
    #     print("Not yet implemented")
    # Case 2: interpolation
    # get the integer of how many elements the new t_grid will have
    n_new_t_grid = int(t_span_in_h/temp_res_in_h)
    # get the new t_grid
    new_t_grid = grids_dict['t_grid'][0] + np.arange(n_new_t_grid + 1) * (t_span_in_h/n_new_t_grid) * 3600
    # perform the 1D interpolations
    new_u_data = interpolate.interp1d(grids_dict['t_grid'], u_data, axis=0, kind='linear')(new_t_grid)
    new_v_data = interpolate.interp1d(grids_dict['t_grid'], v_data, axis=0, kind='linear')(new_t_grid)
    # return new t_grid and values
    return new_t_grid, new_u_data, new_v_data


# Helper helper functions
def get_abs_time_grid_from_hycom_file(f):
    """Helper function to extract the t_grid in UTC POSIX time from a HYCOM File f."""
    # transform from numpy datetime object to POSIX time
    t_grid = (f.variables['time'].data - np.datetime64(0, 's'))/ np.timedelta64(1, 's')
    return t_grid


def grids_interval_sanity_check(grids_dict, lat_interval, lon_interval, t_interval):
    """Advanced Check for warning of partially being out of bound in space or time."""
    # collateral check
    if len(grids_dict['x_grid']) == 0 or len(grids_dict['y_grid']) == 0:
        raise ValueError("None of the requested spatial area is in the file.")
    if len(grids_dict['t_grid']) == 0:
        raise ValueError("None of the requested t_interval is in the file.")

    # data partially not in it check
    if grids_dict['y_grid'][0] > lat_interval[0] or grids_dict['y_grid'][-1] < lat_interval[1]:
        warnings.warn("Part of the lat requested area is outside of file.", RuntimeWarning)
    if grids_dict['x_grid'][0] > lon_interval[0] or grids_dict['x_grid'][-1] < lon_interval[1]:
        warnings.warn("Part of the lon requested area is outside of file.", RuntimeWarning)
    if grids_dict['t_grid'][0] > t_interval[0].timestamp():
        raise ValueError("The starting time t_interval[0] is not in the file.")
    if t_interval[1] is not None:
        if grids_dict['t_grid'][-1] < t_interval[1].timestamp():
            warnings.warn("Part of the requested time is outside of file.", RuntimeWarning)


def get_grid_dict_from_file(file):
    """Helper function to create a grid dict from a local nc3 file."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    f = xr.open_dataset(file)
    # get the time coverage in POSIX
    t_grid = get_abs_time_grid_from_hycom_file(f)
    y_range = [f.variables['lat'].data[0], f.variables['lat'].data[-1]]
    x_range = [f.variables['lon'].data[0], f.variables['lon'].data[-1]]
    # close netCDF file
    f.close()
    # create dict
    return {"t_range": [datetime.fromtimestamp(t_grid[0], timezone.utc),
                           datetime.fromtimestamp(t_grid[-1], timezone.utc)],
            "y_range": y_range,
            "x_range": x_range}


def add_element_front_and_back_if_possible(grid, grid_inds):
    """Helper function to add the elements front and back of the indicies if possible."""
    # insert in the front if there's space
    if grid_inds[0] > 0:
        grid_inds = np.insert(grid_inds, 0, grid_inds[0] - 1, axis=0)
    # insert in the end if there's space
    if grid_inds[-1] < len(grid) - 1:
        grid_inds = np.insert(grid_inds, len(grid_inds), grid_inds[-1] + 1, axis=0)
    return grid_inds