import casadi as ca
import datetime
import netCDF4
import numpy as np


def get_current_data_subset(nc_file, x_0, x_T, deg_around_x0_xT_box, fixed_time_index=None):
    """ Function to read a subset of the nc_file current data bounded by a box spanned by the x_0 and x_T points.
    Inputs:
        nc_file                 path to nc file
        x_0, x_T                [lon, lat] locations
        deg_around_x0_xT_box    buffer around the box in degrees
        fixed_time_index        index of the data for returning a fixed-time current field
    """
    f = netCDF4.Dataset(nc_file)
    # TODO: implement automatic time-indexing if x_0 contains a time
    # make sure we only take the lat, lon from x_0, x_T
    x_0 = x_0[:2]
    x_T = x_T[:2]

    # Step 1: get the grid
    xgrid = f.variables['lon'][:]
    ygrid = f.variables['lat'][:]
    t_grid = f.variables['time'][:]

    # TODO: implement time-zone awareness (will be relevant for solar charging)
    try:
        time_origin = datetime.datetime.strptime(f.variables['time'].__dict__['time_origin'],
                                                 '%Y-%m-%d %H:%M:%S')
    except:
        time_origin = datetime.datetime.strptime(f.variables['time'].__dict__['units'],
                                                 'hours since %Y-%m-%d %H:%M:%S.000 UTC')

    n_steps = t_grid.shape[0]

    # Step 2: find the sub-setting
    lon_bnds = [min(x_0[0], x_T[0]) - deg_around_x0_xT_box, max(x_0[0], x_T[0]) + deg_around_x0_xT_box]
    lat_bnds = [min(x_0[1], x_T[1]) - deg_around_x0_xT_box, max(x_0[1], x_T[1]) + deg_around_x0_xT_box]

    # get indices
    ygrid_inds = np.where((ygrid > lat_bnds[0]) & (ygrid < lat_bnds[1]))[0]
    xgrid_inds = np.where((xgrid > lon_bnds[0]) & (xgrid < lon_bnds[1]))[0]

    # Step 1.5 print what will be done
    if fixed_time_index is None:
        print("Input Fieldset from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
            start=(time_origin + datetime.timedelta(hours=t_grid[0])).strftime('%Y-%m-%d %H:%M:%S'),
            end=(time_origin + datetime.timedelta(hours=t_grid[-1])).strftime('%Y-%m-%d %H:%M:%S'),
            n_steps=n_steps, time=(t_grid[1] - t_grid[0])))
        slice_for_time_dim = np.s_[:]
    else:
        print("Fieldset fixed time at: {time}".format(
            time=(time_origin + datetime.timedelta(hours=t_grid[fixed_time_index])).strftime('%Y-%m-%d %H:%M:%S')))
        slice_for_time_dim = np.s_[fixed_time_index]

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
    # zero-baseline the t_grid because the interpolation function used by the simulator and ipopt requires it
    # TODO: this needs to change when we need to do replanning that is not at aligned with the fieldset times!
    t_grid = (t_grid - t_grid[0]) * 3600.
    grids_dict = {'xgrid': xgrid[xgrid_inds], 'ygrid': ygrid[ygrid_inds], 't_grid': t_grid, 'time_origin': time_origin}

    #TODO: we replace the masked array with fill value 0 because otherwise interpolation doesn't work.
    # Though that means we cannot anymore detect if we're on land or not (need a way to do that/detect stranding)
    return grids_dict, u_data.filled(fill_value=0.), v_data.filled(fill_value=0.)


def get_interpolation_func(grids_dict, u_data, v_data, type='bspline', fixed_time_index=None):
    # fit interpolation function
    if fixed_time_index is None:    # time varying current
        # U field varying
        u_curr_func = ca.interpolant('u_curr', type,
                                     [grids_dict['t_grid'], grids_dict['ygrid'], grids_dict['xgrid']],
                                     u_data.ravel(order='F'))
        # V field varying
        v_curr_func = ca.interpolant('v_curr', type,
                                     [grids_dict['t_grid'], grids_dict['ygrid'], grids_dict['xgrid']],
                                     v_data.ravel(order='F'))
    else:      # fixed current
        # U field fixed
        u_curr_func = ca.interpolant('u_curr', type, [grids_dict['ygrid'], grids_dict['xgrid']], u_data.ravel(order='F'))
        # V field fixed
        v_curr_func = ca.interpolant('v_curr', type, [grids_dict['ygrid'], grids_dict['xgrid']], v_data.ravel(order='F'))

    return u_curr_func, v_curr_func


def get_interpolation_func_from_fieldset(fieldset, type='bspline', fixed_time_index=None):
    # Step 1: get grid
    xgrid = fieldset.U.lon
    ygrid = fieldset.U.lat
    t_grid = fieldset.U.grid.time
    n_steps = t_grid.shape[0]
    if fixed_time_index is None:    # time varying current
        print("Fieldset from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
            start=fieldset.U.grid.time_origin, end=str(fieldset.U.grid.time_origin.fulltime(t_grid[-1])),
            n_steps=n_steps, time=t_grid[1] / 3600))

        # Step 2: extract field data
        # [tdim, zdim, ydim, xdim]
        if len(fieldset.U.data.shape) == 4:  # if there is a depth dimension in the dataset
            u_data = fieldset.U.data[:, 0, :, :]
            v_data = fieldset.V.data[:, 0, :, :]
        # [tdim, ydim, xdim]
        elif len(fieldset.U.data.shape) == 3:  # if there is no depth dimension in the dataset
            u_data = fieldset.U.data[:, :, :]
            v_data = fieldset.V.data[:, :, :]

        # U field varying
        u_curr_func = ca.interpolant('u_curr', type, [t_grid, ygrid, xgrid], u_data.ravel(order='F'))
        # V field varying
        v_curr_func = ca.interpolant('v_curr', type, [t_grid, ygrid, xgrid], v_data.ravel(order='F'))
    else:      # fixed current
        print("Fieldset fixed time at: {time}".format(
            time=str(fieldset.U.grid.time_origin.fulltime(t_grid[fixed_time_index]))))

        # Step 2: extract field data
        # [tdim, zdim, ydim, xdim]
        if len(fieldset.U.data.shape) == 4:  # if there is a depth dimension in the dataset
            u_data = fieldset.U.data[fixed_time_index, 0, :, :]
            v_data = fieldset.V.data[fixed_time_index, 0, :, :]
        # [tdim, ydim, xdim]
        elif len(fieldset.U.data.shape) == 3:  # if there is no depth dimension in the dataset
            u_data = fieldset.U.data[fixed_time_index, :, :]
            v_data = fieldset.V.data[fixed_time_index, :, :]
        else:
            raise ValueError("Input Fieldset does not have 3 or 4 dimensions.")

        # U field fixed
        u_curr_func = ca.interpolant('u_curr', type, [ygrid, xgrid], u_data.ravel(order='F'))
        # V field fixed
        v_curr_func = ca.interpolant('v_curr', type, [ygrid, xgrid], v_data.ravel(order='F'))

    return u_curr_func, v_curr_func