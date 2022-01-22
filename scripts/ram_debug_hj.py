import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import time
import netCDF4
# file = '/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/data_archive/analytical_currents/anl_dg.nc'
file = '/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/data_archive/analytical_currents/benchmark_qdg.nc'

@profile
def main():
    # init settings
    x_init = [0.85, 0.2]
    x_target = [0.8, 0.5]
    r = 0.05

    # general settings
    # for backwards reachability
    times = np.linspace(0, -10, 20)
    uMode = 'min'
    # # for forward reachability
    # times = np.linspace(0, 20, 30)
    # uMode = 'max'

    grid = hj.Grid.from_grid_definition_and_initial_values(hj.sets.Box(lo=np.array([0.01666667, 0.01666667]),
                                                                       hi=np.array([0.98333335, 0.98333335])), (100, 100))
    initial_values = hj.shapes.shape_sphere(grid=grid, center=x_target, radius=r)
    solver_settings = hj.SolverSettings.with_accuracy("high",
                                                      x_init=x_init,
                                                      artificial_dissipation_scheme=
                                                      hj.artificial_dissipation.local_local_lax_friedrichs,
                                                      )
    # run the solver
    Plat2D = hj.systems.Platform2Dcurrents(u_max=0.01, current_file=file, control_mode=uMode,
                                           pre_compute_spatial_interpol=True, spatial_shape=grid.shape)

    print(hj.solver._solve._cache_size())
    terminal_idx, all_values = hj.solver._solve(solver_settings, Plat2D, grid.boundary_conditions,
                                      True, grid.arrays, times, initial_values)

    # change dynamics
    del Plat2D
    Plat2D_new = hj.systems.Platform2Dcurrents(u_max=0.02, current_file=file, control_mode=uMode,
                                           pre_compute_spatial_interpol=True, spatial_shape=grid.shape)
    print(hj.solver._solve._cache_size())
    terminal_idx, all_values = hj.solver._solve(solver_settings, Plat2D_new, grid.boundary_conditions,
                                                True, grid.arrays, times, initial_values)
    print(hj.solver._solve._cache_size())
    print("step cahce: ",hj.solver._step._cache_size())

    hj.solver._solve._clear_cache()
    del hj.solver._solve
    from jax.interpreters import xla
    xla._xla_callable.cache_clear()

if __name__ == '__main__':
    main()


