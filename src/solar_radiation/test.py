import casadi as ca
import numpy as np
import time

import spa

x = ca.MX.sym('x', 2, 2)
A = np.array([[2, 0], [1, 2]])

print(type(A @ x))


#unixtime = ca.MX.sym('t') #time.time()
#lat = ca.MX.sym('x')#37.8715
#lon = ca.MX.sym('y') #-122.2730

# see if the values seem admissible for berkeley at the time of execution
unixtime = time.time() #1635151480.8374586 + 3600 * 8
lat = 37.8715
lon = -122.2730

elev = 0

pressure = 1013.25
temp = 12
delta_t = 67.0
atmos_refract = 0.5667
numthreads = 0
#print(spa.solar_position_numpy(unixtime, lat, lon, elev, pressure, temp, delta_t,
#                         atmos_refract, numthreads, sst=False, esd=False)[3])
print(spa.solar_rad(unixtime, lat, lon))