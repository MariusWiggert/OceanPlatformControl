# OceanPlatformControl Project

## Setup of environment

The libraries we use ([ocean parcels](http://oceanparcels.org), [casadi](http://casadi.org)) all have a C++ backend to make them faster, 
hence we can’t just simply use virtual environments and have to do some other package installing as well.

**Option 1: Docker**
Pro: easy to achieve consistency across devices -> eventually for putting on the hardware
Cons: harder to modify the system library files (shouldn’t be the case very often)...

Instructions: TBD

**Option 2: Local Conda environment** 
Pro: easier to add libraries and experiment around with the code
Cons: less consistency across multiple users

Instructions:
1. Install Anaconda OR miniconda (only one of them) [Link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
1. Clone project repo: 
`git clone https://github.com/MariusWiggert/OceanPlatformControl.git`
1. In the terminal, navigate inside the project repo and run the install bash script
`source ./setup/set_up_ocean_platform_env.sh`
This will create a new conda environment called `ocean_platform` which contains all that is needed for now.










