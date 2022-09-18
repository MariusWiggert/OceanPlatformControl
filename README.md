# OceanPlatformControl Project

## Setup of environment
In any case install conda:
Install Anaconda OR miniconda (only one of them) [Link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

### If you only develop on OceanPlatformControl
```sh
# 1. Clone project repo: 
git clone https://github.com/MariusWiggert/OceanPlatformControl.git
# 2. In the terminal, navigate inside the project repo and run the install bash script
cd setup/
source  set_up_ocean_platform_env.sh
```

### Minimal setup
Recreate the minimal setup of Marius with hj_reachability (v0.4).
```sh
cd setup/
conda create -y -n ocean_minimal python=3.9.11
conda activate ocean_minimal
conda install -y -c jupyter cartopy ffmpeg
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python
pip install -r requirements_minimal.txt
# hj_reachability needs to be installed after jax
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git
```
### If you develop on OceanPlatformControl and hj_reachability simulatenously

3. Now you need to download our two repos and put them in the same top_folder
* top_folder
    * OceanPlatformControl
    * hj_reachability_c3
2. Clone project repo: 
`git clone https://github.com/MariusWiggert/OceanPlatformControl.git`
3. Clone hj_reachability repo:
`git clone https://github.com/MariusWiggert/hj_reachability_c3`
4. In the terminal, navigate inside the project repo, open the script:
`set_up_ocean_platform_env.sh`
Comment out line 13 (where hj_reachability is installed)
Now run the script `source set_up_ocean_platform_env.sh`


This will create a new conda environment called `ocean_platform` which contains all that is needed.

### Data

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
├── setup.py
└── tmp
```
</details>

- Data can also be downloaded using [download_files_from_c3_to_local.py](scripts/tutorial/data_sources/download_files_from_c3_to_local.py) from c3. There is a better way to download it created by Jonas which is not yet merged.
- There is other data available to download that was used in previous experiments (e.g. analytical_currents are samples where it is possible to calculate the currents everywhere because you know the physical model, e.g. of a double gyre).
- Both Hycom and Copernicus use a resolution of 1/12th degree (~7km).
- Hycom uses 1/25th degree in the Gulf of Mexico (GOM).
