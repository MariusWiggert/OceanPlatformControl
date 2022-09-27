import os, psutil
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
import gc

pid = os.getpid()
process = psutil.Process()

print(f"PID {pid}")

print(f"Memory initially: {process.memory_info().rss/1e6:.1f}MB")

arena = ArenaFactory.create(
    scenario_name="gulf_of_mexico_HYCOM_forecast_Copernicus_hindcast", pid=pid, timing=True
)

print(f"Memory before collection: {process.memory_info().rss/1e6:.1f}MB")

# del arena.platform
# del arena.ocean_field
del arena
gc.collect()

print(f"Memory after collection: {process.memory_info().rss/1e6:.1f}MB")
