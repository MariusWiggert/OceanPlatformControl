from c3python import C3Python # https://github.com/c3aidti/c3python
c3 = C3Python(
    url='https://dev01-seaweed-control.c3dti.ai',
    tenant='seaweed-control',
    tag='dev01',
    keyfile='/Users/jeromejeannin/Library/CloudStorage/GoogleDrive-jeromemjeannin@gmail.com/My Drive/Master Thesis/OcceanPlatformControl/setup/c3-rsa',
    username='jeanninj@berkeley.edu',
).get_c3()

controllerSetting = c3.ControllerSetting.get('Short_Horizon_CopernicusGT_StraightLine')
simConfig = c3.Experiment.get("Short_Horizon_CopernicusGT").simConfig
missions = c3.Mission.fetch(spec={'filter': 'experiment.id=="Short_Horizon_CopernicusGT"'})