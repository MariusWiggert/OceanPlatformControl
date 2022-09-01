# OceanPlatformControl Project

## Setup of environment
In any case install conda:
Install Anaconda OR miniconda (only one of them) [Link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

### If you only develop on OceanPlatformControl

**FOR Mac M1:**

1. Clone project repo: 
`git clone https://github.com/MariusWiggert/OceanPlatformControl.git`
2. In the terminal, navigate inside the project repo and run the install bash script of the `m1` folder
`chmod 700 m1/setup_m1.sh`
`python3 m1/setup_m1.sh`
3. then execute the setup.py in the repo folder with pip install
   `pip install -e .`

### If you develop on OceanPlatformControl and hj_reachability simulatenously

1. Now you need to download our two repos and put them in the same top_folder
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










