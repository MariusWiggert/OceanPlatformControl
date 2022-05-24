from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector


def get_error_ocean_current_vector(forecast: OceanCurrentVector,
                                   true_current: OceanCurrentVector) -> OceanCurrentVector:
    return forecast.subtract(true_current)


def get_improved_forecast(forecasts: float, errors: float) -> float:
    return forecasts - errors
