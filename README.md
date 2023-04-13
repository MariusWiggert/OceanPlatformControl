# OceanPlatformControl Project

## Setup of environment
<details>
<summary> Minimal setup if you only develop on OceanPlatformControl</summary>
Install Anaconda OR miniconda (only one of them), more  [information](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). Recreate the minimal setup of Marius with hj_reachability (v0.4).

   ```sh
# 1. Clone project repo: 
git clone https://github.com/MariusWiggert/OceanPlatformControl.git
```
**LINUX** (might also work on x86 macs but not tested yet)

First create a conda environment

```sh
conda create -y -n ocean_minimal python=3.9.15
conda activate ocean_minimal
```

Then run the installation script (you might have to run `chmod 700 setup/install.sh` before):

```sh
./setup/install.sh
```



**MAC M1 ARM**

First create a conda environment

```sh
conda create -y -n ocean_minimal python=3.9
conda activate ocean_minimal
```

Then run the installation script (you might have to run `chmod 700 setup/m1_install.sh` before):

```sh
./setup/m1_install.sh
```

**WINDOWS**

JAX depends on XLA which needs to be installed as the jaxlib package, which is (officialy) only supported on Linux (Ubuntu) and macOS for now,
according to the github https://github.com/google/jax

Fix: Install Ubuntu on *Windows Subsystem for Linux (WSL)* by oppening PowerShell as administrator and run:
```sh
wsl --install
```
which will enable all the features necessary to run WSL and install the Ubuntu distribution.
Alternatively, follow:
https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview

Visual Studio Code is well suited for developing in WSL and makes the integration easy, see: 
https://code.visualstudio.com/docs/remote/wsl

For visualization of figures, Windows 11 (> Build 22000) supports running Linux GUI apps and allows interactive plotting.
For Windows 10, no direct GUI support which means that plt.show() for example won't work directly. 
Other solutions exist, such as saving the figure or workarounds provided by the community.

Once you have WSL installed and running, install conda or miniconda and follow the same instructions as for the *minimal setup for Linux*.
</details>
<details>
<summary> If you develop on OceanPlatformControl and hj_reachability simulatenously </summary>

1. Now you need to download our two repos and put them in the same top_folder
* top_folder
    * OceanPlatformControl
    * hj_reachability_c3
2. Clone project repo: 
`git clone https://github.com/MariusWiggert/OceanPlatformControl.git`
3. Clone hj_reachability repo:
`git clone https://github.com/MariusWiggert/hj_reachability_c3`
4. In the terminal, navigate inside the project repo, create the conda environment (see minimal instructions), open the script:
`setup/install.sh` or its m1 version.
Comment out the line where hj_reachability is installed.
Now run the script as described above.
5. Move into the hj_reachability_c3 directory. Within your activated conda environment, run `pip install -e .`


This will create a new conda environment called `ocean_platform` which contains all that is needed.
</details>

## Data

Download the data (manually) from the google drive (Ocean_Platform_Control/3_data/data) and unzip the data in a directory called `data` in this project. The data that you need is `forecast_test`, `hindcast_test`, `nutrients`. Your directory structure should look similar to the following: 
<details>
<summary> Directory structure </summary>

```sh 
OceanPlatformControl/
├── config
|── data
│   ├── forecast_test
│   ├── hindcast_test
│   ├── nutrients
├── generated_media # Created automatically
│   ├── currents_animation.mp4
│   ├── solar_animation.mp4
│   ├── solar_test_animation.mp4
│   └── test_hindcast_current_animation.mp4
├── LICENSE
├── models
├── ocean_navigation_simulator
├── README.md
├── scripts
├── setup
├── tests
├── setup.py
└── tmp
```
</details>

- Data can also be downloaded using [download_files_from_c3_to_local.py](scripts/tutorial/data_sources/download_files_from_c3_to_local.py) from c3. There is a better way to download it created by Jonas which is not yet merged.
- There is other data available to download that was used in previous experiments (e.g. analytical_currents are samples where it is possible to calculate the currents everywhere because you know the physical model, e.g. of a double gyre).
- Both Hycom and Copernicus use a resolution of 1/12th degree (~7km).
- Hycom uses 1/25th degree in the Gulf of Mexico (GOM).


### Testing your installation 
For testing your installation you can run the tutorial scripts (make sure you run them from the root repo folder and the folder generated_media exists in the root repo folder), i.e:
```sh
python3 scripts/tutorial/controller/hj_planner.py
```
## Development
<details>
<summary>How to use our development tools (testing, styling, make)</summary>
To install the development packages (tests, linting, etc.), run

```sh
python3 -m pip install -e ".[dev]"
```
### Styling and formatting
```sh
black . # In place formatter, adheres mostly to PEP8
flake8  # Code linter with stylistic conventions adhering to PEP8
isort . # Sorts and formats imports in python files
```

### Tests
Testing with pytest, this is how you can run different levels of granularity of tests.
```sh
python3 -m pytest # all tests
python3 -m pytest tests/food # tests under a directory
python3 -m pytest tests/food/test_fruits.py  # tests for a single file
python3 -m pytest tests/food/test_fruits.py::test_is_crisp  # tests for a single function
# Get test coverage for report
# Runs all tests, results will be in htmlcov/index.html
python3 -m pytest --cov ocean_navigation_simulator --cov-report html
# To exclude code from the coverage report, add these lines
# pragma: no cover, <MESSAGE>
```

### Make commands
You can use `make` to execute targets defined in the `Makefile`. If make is not installed, run `sudo apt install make`.
```sh
# See available make commands
make help
# Execute several clean commands
make clean
# Execute all non-training tests
make test
# Execute style formatting
make style
```
### Configs
The development tools are configured in the following files. While trying to adhere to standards, we made some exceptions and ignored some directories.
```sh
.flake8 # flake8
pyproject.toml # black, isort, pytest
Makefile # make
```
</details>

### IDE settings (VScode, Pycharm)
You can also change settings in your IDE to do auto formatting (black) on save and run flake8 on save if you do not want to run them via terminal. Here are articles to setup [black on Pycharm](https://akshay-jain.medium.com/pycharm-black-with-formatting-on-auto-save-4797972cf5de), [black on VScode](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0), [flake8 on VScode](https://code.visualstudio.com/docs/python/linting).
