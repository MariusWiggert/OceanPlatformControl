from ocean_navigation_simulator.generative_error_model.data_files.BuoyData import BuoyDataSofar, BuoyDataCopernicus
import yaml


def test_sofar():
    config_path = "../../../scenarios/generative_error_model/config_sofar_data.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sofar_data = BuoyDataSofar(config)
    return sofar_data.data


def test_copernicus():
    config_path = "../../../scenarios/generative_error_model/config_buoy_data.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cop_data = BuoyDataCopernicus(config)
    return cop_data.data


if __name__ == "__main__":
    sofar_data = test_sofar()
    print(sofar_data.describe())
    import matplotlib.pyplot as plt

    buoy_names = list(set(sofar_data["buoy"]))
    colors = {buoy: num for buoy, num in zip(buoy_names, range(len(buoy_names)))}
    plt.scatter(sofar_data["lon"], sofar_data["lat"], c=sofar_data["buoy"].map(colors))
    plt.show()
