import casadi as ca

def get_interpolation_func(fieldset, conv_m_to_deg, type='bspline', fixed_time_index=None):

    # Step 1: get grid
    xgrid = fieldset.U.lon
    ygrid = fieldset.U.lat
    t_grid = fieldset.U.grid.time
    n_steps = t_grid.shape[0]
    if n_steps == 1:    # when the fieldset itself only has one time index
        fixed_time_index = 0
    if fixed_time_index is None:    # time varying current
        print("Fieldset from {start} to {end} in {n_steps} time steps of {time:.2f} hour(s) resolution".format(
            start=fieldset.U.grid.time_origin, end=str(fieldset.U.grid.time_origin.fulltime(t_grid[-1])),
            n_steps=n_steps, time=t_grid[1] / 3600))

        # Step 2: extract field data and convert to deg/sec
        # [tdim, zdim, ydim, xdim]
        if len(fieldset.U.data.shape) == 4:  # if there is a depth dimension in the dataset
            u_data = fieldset.U.data[:, 0, :, :] / conv_m_to_deg
            v_data = fieldset.V.data[:, 0, :, :] / conv_m_to_deg
        # [tdim, ydim, xdim]
        elif len(fieldset.U.data.shape) == 3:  # if there is no depth dimension in the dataset
            u_data = fieldset.U.data[:, :, :] / conv_m_to_deg
            v_data = fieldset.V.data[:, :, :] / conv_m_to_deg

        # U field varying
        u_curr_func = ca.interpolant('u_curr', type, [t_grid, ygrid, xgrid], u_data.ravel(order='F'))
        # V field varying
        v_curr_func = ca.interpolant('v_curr', type, [t_grid, ygrid, xgrid], v_data.ravel(order='F'))
    else:      # fixed current
        print("Fieldset fixed time at: {time}".format(
            time=str(fieldset.U.grid.time_origin.fulltime(t_grid[fixed_time_index]))))

        # Step 2: extract field data and convert to deg/sec
        # [tdim, zdim, ydim, xdim]
        if len(fieldset.U.data.shape) == 4:  # if there is a depth dimension in the dataset
            u_data = fieldset.U.data[fixed_time_index, 0, :, :] / conv_m_to_deg
            v_data = fieldset.V.data[fixed_time_index, 0, :, :] / conv_m_to_deg
        # [tdim, ydim, xdim]
        elif len(fieldset.U.data.shape) == 3:  # if there is no depth dimension in the dataset
            u_data = fieldset.U.data[fixed_time_index, :, :] / conv_m_to_deg
            v_data = fieldset.V.data[fixed_time_index, :, :] / conv_m_to_deg

        # U field fixed
        u_curr_func = ca.interpolant('u_curr', type, [ygrid, xgrid], u_data.ravel(order='F'))
        # V field fixed
        v_curr_func = ca.interpolant('v_curr', type, [ygrid, xgrid], v_data.ravel(order='F'))

    return u_curr_func, v_curr_func