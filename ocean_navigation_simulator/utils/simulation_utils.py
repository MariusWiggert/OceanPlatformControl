import casadi as ca
from datetime import datetime, timedelta
import netCDF4
import numpy as np
import math
import warnings


def convert_to_lat_lon_time_bounds(x_0, x_T, deg_around_x0_xT_box, temp_horizon_in_h):
    """
    We want to switch over to using lat-lon bounds but a lot of functions in the code already use x_0 and x_T with the
    deg_around_x0_xT_box, so for now use this to convert.
    Args:
        x_0: [lat, lon, charge, timestamp]
        x_T: [lon, lat] goal locations
        deg_around_x0_xT_box: buffer around the box in degrees
        temp_horizon_in_h: maximum temp_horizon to look ahead of x_0 time in hours

    Returns:
        t_interval: if time-varying: [t_0, t_T] in POSIX time
                                     where t_0 and t_T are the start and end timestamps respectively
                    if fixed_time:   [fixed_timestamp] in POSIX
        lat_bnds: [y_lower, y_upper] in degrees
        lon_bnds: [x_lower, x_upper] in degrees
    """
    if temp_horizon_in_h is None:
        t_interval = [x_0[3], None]
    else:
        t_interval = [x_0[3], x_0[3] + temp_horizon_in_h * 3600]
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


# C3 file based data-access function
def get_current_data_subset_from_c3_file(
        t_interval,
        lat_interval,
        lon_interval,
):
    """ Function to get a subset of Hindcast current data via the C3 data integration.
    For description of input and outputs see the get_current_data_subset function.
    """

    # Step 1: get required file references and data from C3 file DB
    # Step 1.1: Getting time and formatting for the db query
    start = datetime.utcfromtimestamp(t_interval[0])
    end = datetime.utcfromtimestamp(t_interval[1])

    # Step 1.2: Getting correct range of nc files from database
    filter_string = 'start>=' + '"' + start.strftime("%Y-%m-%d") + '"' + \
                    ' && end<=' + '"' + end.strftime("%Y-%m-%d") + "T23:00:00.000" + '"' \
                                                                                     ' && status==' + '"' + 'downloaded' + '"'
    objs_list = c3.HindcastFile.fetch({'filter': filter_string, "order": "start"}).objs

    # some basic sanity checks
    if objs_list is None:
        raise ValueError("No files in the database for the selected t_interval")
    if len(objs_list) != (end - start).days + 1:
        raise ValueError("DB Query didn't return the expected number of files (one per day), check DB and code.")

    # Step 1.3: extract url and start list from the query results
    urls_list = [obj.file.url for obj in objs_list]
    start_list = [obj.start for obj in objs_list]

    # Step 2: Prepare the stacking loop by getting the x, y grids and subsetting indices in x, y
    # Note: these stay constant across files in this case where all files have same lat-lon range

    # Step 2.1: open the file and get the x and y grid
    f = c3.HycomUtil.nc_open(urls_list[0])
    xgrid = f.variables['lon'][:].data
    ygrid = f.variables['lat'][:].data

    # Step 2.2: get the respective indices of the lat, lon subset from the file grids
    ygrid_inds = np.where((ygrid >= lat_interval[0]) & (ygrid <= lat_interval[1]))[0]
    xgrid_inds = np.where((xgrid >= lon_interval[0]) & (xgrid <= lon_interval[1]))[0]

    # Step 2.3 initialze t_grid stacking variable
    full_t_grid = []

    # Step 3: iterate over all files in order and stack the current data and absolute t_grids
    for idx in range(len(start_list)):
        # Step 3.0: load the current data file
        f = c3.HycomUtil.nc_open(urls_list[idx])
        # set the default start and end time
        start_hr, end_hr = 0, 24

        # Step 3.1: do the time-subsetting
        # Case 1: file is first -- get data from the file from the hour before or at t_0
        if idx == 0:
            start_hr = math.floor((t_interval[0] - start_list[idx].timestamp()) / 3600)
        # Case 2: file is last -- get data from file until or after the hour t_T
        if idx == len(start_list) - 1:
            end_hr = math.ceil((t_interval[1] - start_list[idx].timestamp()) / 3600) + 1

        # Step 3.2: extract data from the file
        u_data = f.variables['water_u'][start_hr:end_hr, 0, ygrid_inds, xgrid_inds]
        v_data = f.variables['water_v'][start_hr:end_hr, 0, ygrid_inds, xgrid_inds]

        # Step 3.3: stack the sub-setted abs_t_grid and current data
        full_t_grid = full_t_grid + [start_list[idx].timestamp() + i * 3600 for i in range(start_hr, end_hr)]

        if idx == 0:
            full_u_data = u_data
            full_v_data = v_data
        else:
            full_u_data = np.concatenate((full_u_data, u_data), axis=0)
            full_v_data = np.concatenate((full_v_data, v_data), axis=0)

    # Step 4: create dict to output
    grids_dict = {'x_grid': xgrid[xgrid_inds], 'y_grid': ygrid[ygrid_inds], 't_grid': full_t_grid}

    # Step 5: # log what data has been subsetted
    print("Subsetted data from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
        start=datetime.utcfromtimestamp(grids_dict['t_grid'][0]).strftime('%Y-%m-%d %H:%M:%S UTC'),
        end=datetime.utcfromtimestamp(grids_dict['t_grid'][-1]).strftime('%Y-%m-%d %H:%M:%S UTC'),
        n_steps=len(grids_dict['t_grid']), time=(grids_dict['t_grid'][1] - grids_dict['t_grid'][0]) / 3600.))

    # Step 6: return the grids_dict and the stacked data
    # TODO: currently, we just do fill_value =0 but then we can't detect if we're on land.
    # We need a way to do that in the simulator, doing it via the currents could be one way.
    return grids_dict, full_u_data.filled(fill_value=0.), full_v_data.filled(fill_value=0.)


# define overarching loading and sub-setting function
# define overarching function
def get_current_data_subset(t_interval, lat_interval, lon_interval,
                        data_type, access,
                        file = None, C3_hindcast_max_temp_in_h = 120):
    """ Function to get a subset of current data either via local file or via C3 database of files.
    Inputs:
        t_interval              if time-varying: [t_0, t_T] in POSIX time
                                where t_0 and t_T are the start and end timestamps respectively
                                if t_T is None and we use a file, the full available time is returned.
        lat_interval            [y_lower, y_upper] in degrees
        lon_interval            [x_lower, x_upper] in degrees

        data_type               string either 'F' or 'H'. Specifies if Forecast for Hindcast data is requested.
        access                  string either 'local' or 'C3'. Specifies if a local file or C3 database is used.
        file                    string of the path to the file (required for all local and C3 Forecasts but not H)

    Outputs:
        grids_dict              dict containing x_grid, y_grid, t_grid
        u_data                  [T, Y, X] matrix of the ocean currents in x direction in m/s
        v_data                  [T, Y, X] matrix of the ocean currents in y direction in m/s
    """
    # Step 0: check if its C3 and Hindcasts, then we need extra function
    if data_type == 'H' and access == 'C3':
        if t_interval[1] is None:
            t_interval[1] = t_interval[0] + C3_hindcast_max_temp_in_h * 3600
        grids_dict, u_data, v_data = get_current_data_subset_from_c3_file(t_interval, lat_interval, lon_interval)
        return grids_dict, u_data, v_data

    # Step 1: open the file
    # Check if C3 or not
    if access == 'C3':
        f = c3.HycomUtil.nc_open(file)
    elif access == 'local':
        f = netCDF4.Dataset(file)

    # Step 2: Extract the grids
    x_grid = f.variables['lon'][:].data
    y_grid = f.variables['lat'][:].data
    t_grid = get_abs_time_grid_for_hycom_file(f, data_type)

    # Step 3: get the subsetting indices for space and time
    ygrid_inds = np.where((y_grid >= lat_interval[0]) & (y_grid <= lat_interval[1]))[0]
    ygrid_inds = add_element_front_and_back_if_possible(y_grid, ygrid_inds)
    xgrid_inds = np.where((x_grid >= lon_interval[0]) & (x_grid <= lon_interval[1]))[0]
    xgrid_inds = add_element_front_and_back_if_possible(x_grid, xgrid_inds)
    if t_interval[1] is None: # take full file time contained in the file
        tgrid_inds = np.where((t_grid >= t_interval[0]))[0]
    else: # subset ending time
        tgrid_inds = np.where((t_grid >= t_interval[0]) & (t_grid <= t_interval[1]))[0]
    tgrid_inds = add_element_front_and_back_if_possible(t_grid, tgrid_inds)

    # Step 3.1: create grid and use that as sanity check if any relevant data is contained in the file
    try:
        grids_dict = {'x_grid': x_grid[xgrid_inds], 'y_grid': y_grid[ygrid_inds],
                      't_grid': t_grid[tgrid_inds]}
    except:
         raise ValueError("None of the requested data contained in file. Check File.")

    # Step 3.2: Advanced sanity check if only partial area is contained in file
    grids_interval_sanity_check(grids_dict, lat_interval, lon_interval, t_interval)

    # Step 4: extract data
    # Note: HYCOM is [tdim, zdim, ydim, xdim]
    if len(f.variables['water_u'].shape) == 4:  # if there is a depth dimension in the dataset
        u_data = f.variables['water_u'][tgrid_inds, 0, ygrid_inds, xgrid_inds]
        v_data = f.variables['water_v'][tgrid_inds, 0, ygrid_inds, xgrid_inds]
    else:
        raise ValueError("Current data in nc file does not have 4 dimensions. Check file.")

    print("Subsetted data from {start} to {end} in {n_steps} time steps".format(
        start=datetime.utcfromtimestamp(grids_dict['t_grid'][0]).strftime('%Y-%m-%d %H:%M:%S UTC'),
        end=datetime.utcfromtimestamp(grids_dict['t_grid'][-1]).strftime('%Y-%m-%d %H:%M:%S UTC'),
        n_steps=len(grids_dict['t_grid'])))

    # TODO: we replace the masked array with fill value 0 because otherwise interpolation doesn't work.
    # Though that means we cannot anymore detect if we're on land or not (need a way to do that/detect stranding)
    return grids_dict, u_data.filled(fill_value=0.), v_data.filled(fill_value=0.)


# Helper functions for the general subset function
def get_abs_time_grid_for_hycom_file(f, data_type):
    """Helper function to extract the t_grid in UTC POSIX time from a HYCOM File f."""
    # Get the t_grid. note that this is in hours from HYCOM data!
    t_grid = f.variables['time'][:]
    # Get the time_origin of the file (Note: this is very tailered for the HYCOM Data)
    try:
        time_origin = datetime.strptime(f.variables['time'].__dict__['time_origin'] + ' +0000',
                                        '%Y-%m-%d %H:%M:%S %z')
    except:
        time_origin = datetime.strptime(f.variables['time'].__dict__['units'] + ' +0000',
                                        'hours since %Y-%m-%d %H:%M:%S.000 UTC %z')

    # for time indexing transform to POSIX time
    abs_t_grid = [(time_origin + timedelta(hours=X)).timestamp() for X in t_grid.data]
    return np.array(abs_t_grid)


def grids_interval_sanity_check(grids_dict, lat_interval, lon_interval, t_interval):
    """Advanced Check for warning of partially being out of bound in space or time."""
    if grids_dict['y_grid'][0] > lat_interval[0] or grids_dict['y_grid'][-1] < lat_interval[1]:
        warnings.warn("Part of the lat requested area is outside of file.", RuntimeWarning)
    if grids_dict['x_grid'][0] > lon_interval[0] or grids_dict['x_grid'][-1] < lon_interval[1]:
        warnings.warn("Part of the lon requested area is outside of file.", RuntimeWarning)
    if t_interval[1] is not None:
        if grids_dict['t_grid'][0] > t_interval[0] or grids_dict['t_grid'][-1] < t_interval[1]:
            warnings.warn("Part of the requested time is outside of file.", RuntimeWarning)


def add_element_front_and_back_if_possible(grid, grid_inds):
    """Helper function to add the elements front and back of the indicies if possible."""
    # insert in the front if possible
    grid_inds = np.insert(grid_inds, 0, max(0, grid_inds[0] - 1), axis=0)
    # insert in the end if possible
    grid_inds = np.insert(grid_inds, len(grid_inds), min(len(grid) - 1, grid_inds[-1] + 1), axis=0)
    return grid_inds