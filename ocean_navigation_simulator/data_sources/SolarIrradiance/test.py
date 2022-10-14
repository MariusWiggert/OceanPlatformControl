import datetime

import matplotlib.pyplot as plt
import numpy as np

import casadi as ca
from ocean_navigation_simulator.data_sources.SolarIrradiance.solar_rad import (
    solar_rad,
)

x = ca.MX.sym("x", 2, 2)
A = np.array([[2, 0], [1, 2]])

print(type(A @ x))

#%%

# unixtime = ca.MX.sym('t') #time.time()
# lat = ca.MX.sym('x')#37.8715
# lon = ca.MX.sym('y') #-122.2730

# see if the values seem admissible for berkeley at the time of execution
# unixtime = time.time() #1635151480.8374586 + 3600 * 8
# lat = 37.8715
# lon = -122.2730
lat, lon = (37.8715, -122.2730)
# lat, lon = (26.0, -87.0)
elev = 0

pressure = 1013.25
temp = 12
delta_t = 67.0
atmos_refract = 0.5667
numthreads = 0
# print(spa.solar_position_numpy(unixtime, lat, lon, elev, pressure, temp, delta_t,
#                         atmos_refract, numthreads, sst=False, esd=False)[3])

#%%
# admissibility issues:
# we
t0 = datetime.datetime(2021, 7, 20)
xt = [t0 + datetime.timedelta(hours=i) for i in range(0, 24)]
yt = [solar_rad(dtime.timestamp(), lat, lon) for dtime in xt]

plt.figure()
plt.title(f"Sunlight over a {t0} at berkeley")
plt.plot(xt, yt)
plt.gcf().autofmt_xdate()
plt.show()
# TODO: more tests of admissibility of this approach

#%% Test 1: calculate solar irridance in Wh for that day and compare with public tables
# http://www.solarelectricityhandbook.com/solar-irradiance.html or others
print("Solar irridance on flat ground from formula in kWh: ", sum(yt) / 1000)
# this returns 3.15 kWh/day but looking at public table for SF it should be ≈6kWh/day.
# => maybe because of the cloud coverage we assume?
#%% Test 2: Check if sunrise-sunset times work out in the local time-zones
# compare with online tables:
# https://www.timeanddate.com/sun/mexico/merida?month=6&year=2021
import pytz

# t0 = datetime.datetime(2021, 11, 6, tzinfo=pytz.timezone('US/Pacific'))
t0 = datetime.datetime(2021, 6, 21, tzinfo=pytz.timezone("UTC"))
# t0 = datetime.datetime(2021, 6, 27, tzinfo=pytz.timezone('America/Merida'))
xt = [t0 + datetime.timedelta(hours=i) for i in range(0, 24)]
yt = [solar_rad(dtime.timestamp(), lat, lon) for dtime in xt]

plt.figure()
plt.title(f"Sunlight over a {t0} at berkeley")
plt.plot(xt, yt)
plt.gcf().autofmt_xdate()
plt.show()
