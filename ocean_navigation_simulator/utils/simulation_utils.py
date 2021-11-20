import casadi as ca
from datetime import datetime, timedelta
import netCDF4
import numpy as np
import bisect
import math

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
    t_interval = [x_0[3], x_0[3] + temp_horizon_in_h * 3600]
    lat_bnds = [min(x_0[1], x_T[1]) - deg_around_x0_xT_box, max(x_0[1], x_T[1]) + deg_around_x0_xT_box]
    lon_bnds = [min(x_0[0], x_T[0]) - deg_around_x0_xT_box, max(x_0[0], x_T[0]) + deg_around_x0_xT_box]

    return t_interval, lat_bnds, lon_bnds


def get_current_data_subset_local_file(
    nc_file,
    t_interval, #temp_res_in_h,   ----> separate function
    lat_bnds, #lat_res_in_deg,
    lon_bnds, #lon_res_in_deg,
    #depth_interval_to_avg_over
):
    """ Function to read a subset of the nc_file current data bounded by a box spanned by the x_0 and x_T points.
    Note: if we want to also include time_subsampling and/or up-sampling we might look into using the function from:
    https://oceanspy.readthedocs.io/en/latest/_modules/oceanspy/subsample.html#cutout
    Inputs:
        nc_file                 path to nc file
        t_interval              if time-varying: [t_0, t_T] in POSIX time
                                where t_0 and t_T are the start and end timestamps respectively
                                if fixed_time:   [fixed_timestamp] in POSIX
        temp_res_in_h           which temporal resolution the time-axis should have
                                e.g. if temp_res_in_h = 1, t_grid = [t_0, t_0 + 3600s, ... t_T]
                                if temp_res_in_h = 5,      t_grid = [t_0, t_0 + 5*3600s, ... t_T]
                                if temp_res_in_h = 0.5,      t_grid = [t_0, t_0 + 1800s, ... t_T]
                                => so either averaging or interpolation needs to be done in the backend
        lat_bnds                [y_lower, y_upper] in degrees
        lat_res_in_deg          which spatial resolution in y direction in degrees
                                e.g. if lat_res_in_deg = 1, y_grid = [y_lower, y_lower + 1, ... y_upper]
                                 => so either averaging or interpolation needs to be done in the backend
        lon_bnds                [x_lower, x_upper] in degrees
        lon_res_in_deg          which spatial resolution in x direction in degrees
                                e.g. if lon_res_in_deg = 1, x_grid = [x_lower, x_lower + 1, ... x_upper]
                                 => so either averaging or interpolation needs to be done in the backend
        depth_interval_to_avg_over
                                Interval to average over the current dimension in meters
                                e.g. [0, 10] then the currents are averaged over the depth 0-10m.

    """
    f = netCDF4.Dataset(nc_file)

    # Step 1: get the grids
    xgrid = f.variables['lon'][:]
    ygrid = f.variables['lat'][:]
    t_grid = f.variables['time'][:] # not this is in hours from HYCOM data!

    # extract time_origin from the files
    try:
        time_origin = datetime.strptime(f.variables['time'].__dict__['time_origin'] + ' +0000',
                                        '%Y-%m-%d %H:%M:%S %z')
    except:
        time_origin = datetime.strptime(f.variables['time'].__dict__['units'] + ' +0000',
                                                 'hours since %Y-%m-%d %H:%M:%S.000 UTC %z')

    # Step 2.1 get the spatial indices
    ygrid_inds = np.where((ygrid > lat_bnds[0]) & (ygrid < lat_bnds[1]))[0]
    xgrid_inds = np.where((xgrid > lon_bnds[0]) & (xgrid < lon_bnds[1]))[0]

    # Step 2.2 Get the temporal indicies
    # for time indexing transform to POSIX time
    abs_t_grid = [(time_origin + timedelta(hours=X)).timestamp() for X in t_grid.data]
    # get the idx of the value left of the demanded time (for interpolation function)
    t_start_idx = bisect.bisect_right(abs_t_grid, t_interval[0]) - 1
    if t_start_idx == len(abs_t_grid) - 1 or t_start_idx == -1:
        raise ValueError("Requested start time is outside of the nc4 file.")

    # get the max time if provided as input
    if t_interval[1] is None:  # all data provided
        t_end_idx = len(abs_t_grid) - 1
    else:
        t_end_idx = bisect.bisect_right(abs_t_grid, t_interval[1])
        if t_end_idx == len(abs_t_grid):
            raise ValueError("nc4 file does not contain requested temporal horizon.")

    slice_for_time_dim = np.s_[t_start_idx:(t_end_idx+1)]

    # Step 2: extract data
    # [tdim, zdim, ydim, xdim]
    if len(f.variables['water_u'].shape) == 4:  # if there is a depth dimension in the dataset
        u_data = f.variables['water_u'][slice_for_time_dim, 0, ygrid_inds, xgrid_inds]
        v_data = f.variables['water_v'][slice_for_time_dim, 0, ygrid_inds, xgrid_inds]
    # [tdim, ydim, xdim]
    elif len(f.variables['water_u'].shape) == 3:  # if there is no depth dimension in the dataset
        u_data = f.variables['water_u'][slice_for_time_dim, ygrid_inds, xgrid_inds]
        v_data = f.variables['water_v'][slice_for_time_dim, ygrid_inds, xgrid_inds]
    else:
        raise ValueError("Current data in nc file has neither 3 nor 4 dimensions. Check file.")

    # create dict
    grids_dict = {'x_grid': xgrid[xgrid_inds], 'y_grid': ygrid[ygrid_inds],
                  't_grid': abs_t_grid[slice_for_time_dim]}

    print("Subsetted data from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
        start=datetime.utcfromtimestamp(grids_dict['t_grid'][0]).strftime('%Y-%m-%d %H:%M:%S UTC'),
        end=datetime.utcfromtimestamp(grids_dict['t_grid'][-1]).strftime('%Y-%m-%d %H:%M:%S UTC'),
        n_steps=len(grids_dict['t_grid']), time=(grids_dict['t_grid'][1] - grids_dict['t_grid'][0])/3600.))

    #TODO: we replace the masked array with fill value 0 because otherwise interpolation doesn't work.
    # Though that means we cannot anymore detect if we're on land or not (need a way to do that/detect stranding)
    return grids_dict, u_data.filled(fill_value=0.), v_data.filled(fill_value=0.)


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


# Functions for using the Data from the C3 Cloud DB

# C3 file based data-access function
def get_current_data_subset_from_c3_file(
        t_interval,  # temp_res_in_h,   ----> separate function
        lat_interval,  # lat_res_in_deg,
        lon_interval,  # lon_res_in_deg,
        # depth_interval_to_avg_over
):
    """ Function to get a subset of current data via the C3 data integration.

    Inputs:
        t_interval              if time-varying: [t_0, t_T] in POSIX time
                                where t_0 and t_T are the start and end timestamps respectively
                                if fixed_time:   [fixed_timestamp] in POSIX
        temp_res_in_h           which temporal resolution the time-axis should have
                                e.g. if temp_res_in_h = 1, t_grid = [t_0, t_0 + 3600s, ... t_T]
                                if temp_res_in_h = 5,      t_grid = [t_0, t_0 + 5*3600s, ... t_T]
                                if temp_res_in_h = 0.5,      t_grid = [t_0, t_0 + 1800s, ... t_T]
                                => so either averaging or interpolation needs to be done in the backend
        lat_interval            [y_lower, y_upper] in degrees
        lat_res_in_deg          which spatial resolution in y direction in degrees
                                e.g. if lat_res_in_deg = 1, y_grid = [y_lower, y_lower + 1, ... y_upper]
                                 => so either averaging or interpolation needs to be done in the backend
        lon_interval            [x_lower, x_upper] in degrees
        lon_res_in_deg          which spatial resolution in x direction in degrees
                                e.g. if lon_res_in_deg = 1, x_grid = [x_lower, x_lower + 1, ... x_upper]
                                 => so either averaging or interpolation needs to be done in the backend
        depth_interval_to_avg_over
                                Interval to average over the current dimension in meters
                                e.g. [0, 10] then the currents are averaged over the depth 0-10m.

    Outputs:
        grids_dict              dict containing x_grid, y_grid, t_grid
        u_data                  [T, Y, X] matrix of the ocean currents in x direction in m/s
        v_data                  [T, Y, X] matrix of the ocean currents in y direction in m/s
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