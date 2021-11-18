import casadi as ca
from datetime import datetime, timedelta
import netCDF4
import numpy as np
import bisect

def convert_to_lat_lon_time_bounds(x_0, x_T, deg_around_x0_xT_box, temp_horizon_in_h=None):
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
    t_interval = [x_0[3], temp_horizon_in_h]
    lat_bnds = [min(x_0[1], x_T[1]) - deg_around_x0_xT_box, max(x_0[1], x_T[1]) + deg_around_x0_xT_box]
    lon_bnds = [min(x_0[0], x_T[0]) - deg_around_x0_xT_box, max(x_0[0], x_T[0]) + deg_around_x0_xT_box]

    return t_interval, lat_bnds, lon_bnds

def get_current_data_subset(
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

    # # extract position & time for the indexing
    # x_0_pos = x_0[:2]
    x_0_posix_time = t_interval[0]
    # x_T = x_T[:2]

    # Step 1: get the grids
    xgrid = f.variables['lon'][:]
    ygrid = f.variables['lat'][:]
    t_grid = f.variables['time'][:] # not this is in hours from HYCOM data!

    try:
        time_origin = datetime.strptime(f.variables['time'].__dict__['time_origin'] + ' +0000',
                                        '%Y-%m-%d %H:%M:%S %z')
    except:
        time_origin = datetime.strptime(f.variables['time'].__dict__['units'] + ' +0000',
                                                 'hours since %Y-%m-%d %H:%M:%S.000 UTC %z')

    # # Step 2: find the sub-setting
    # lon_bnds = [min(x_0_pos[0], x_T[0]) - deg_around_x0_xT_box, max(x_0_pos[0], x_T[0]) + deg_around_x0_xT_box]
    # lat_bnds = [min(x_0_pos[1], x_T[1]) - deg_around_x0_xT_box, max(x_0_pos[1], x_T[1]) + deg_around_x0_xT_box]

    # get indices
    ygrid_inds = np.where((ygrid > lat_bnds[0]) & (ygrid < lat_bnds[1]))[0]
    xgrid_inds = np.where((xgrid > lon_bnds[0]) & (xgrid < lon_bnds[1]))[0]

    # for time indexing transform to POSIX time
    abs_t_grid = [(time_origin + timedelta(hours=X)).timestamp() for X in t_grid.data]
    # get the idx of the value left of the demanded time (for interpolation function)
    t_start_idx = bisect.bisect_right(abs_t_grid, x_0_posix_time) - 1
    if t_start_idx == len(abs_t_grid) - 1 or t_start_idx == -1:
        raise ValueError("Requested subset time is outside of the nc4 file.")

    # get the max time if provided as input
    if t_interval[1] is None:  # all data provided
        t_end_idx = len(abs_t_grid) - 1
    else:
        t_end_idx = bisect.bisect_right(abs_t_grid, x_0_posix_time + t_interval[1]*3600.)
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