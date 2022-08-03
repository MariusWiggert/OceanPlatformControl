import datetime
import time
import os
import shutil

from c3python import C3Python # https://github.com/c3aidti/c3python
import ocean_navigation_simulator.utils.paths as paths

# Create Keypair:
#   openssl genrsa -out c3-rsa 2048
#   openssl rsa -in c3-rsa -pubout > c3-rsa.pub
#   awk '/-END PUBLIC KEY-/ { p = 0 }; p; /-BEGIN PUBLIC KEY-/ { p = 1 }' c3-rsa.pub | tr -d '\n' | pbcopy

# Upload public key to c3:
#   usr = User.get('jeanninj@berkeley.edu')
#   usr.publicKey = 'MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtWyCnoZoSx1mVsVV0Ihh/ivF/+jBGCyqzlPmZ01SISsF8twfSpES2EW4kglKCivvMVwNAga3ljpXH+oIaM93/LZmrZWjhm9BLnHyM2A4Afw80xIHxl8xov/YwSN3N4rZ0W2iqxR7mOJp6I+MuEnkscjn4vKYf05hp3YzuSbNM+diPWF5W1i5NihS0jbNQqqj3ylepvwe5Hs/iqWpQPG1wW3sRgbQZy+yMGZaOVCBrmvewxP7gdzq7jdZrstpQdOTMkUJEuNv2iQOpW9qdKzxLGnsgGJtAIzTiLvBLelA/lvGFPbi2XInYibTtxtZ0KbZQaMJFRH1h31zxM2RB5RZjQIDAQAB'
#   usr.merge()

c3 = C3Python(
    url='https://dev01-seaweed-control.c3dti.ai',
    tenant='seaweed-control',
    tag='dev01',
    keyfile='/Users/jeromejeannin/Library/CloudStorage/GoogleDrive-jeromemjeannin@gmail.com/My Drive/Master Thesis/OcceanPlatformControl/setup/c3-rsa',
    username='jeanninj@berkeley.edu',
).get_c3()


# %%
t_0 = datetime.datetime(2021, 11, 22, 12, 10, tzinfo=datetime.timezone.utc)
x_0 = [-81.74966879179048, 18.839454259572026]
x_T = [-83.17890714569597, 18.946404547127734]
n_days_ahead = 5

# %%


# %%
# Step 1: Find relevant files
fmrc_archive_id = c3.HycomDataArchive.fetch(spec = { 'filter': 'dataset=="GOMu0.04/expt_90.1m000"' }).objs[0].fmrcArchive.id
#%%
start = f'{t_0 - datetime.timedelta(days=1)}'
end = f'{(t_0 + datetime.timedelta(days=n_days_ahead)).replace(hour=23, minute=59, second=0, microsecond=0)}'
files = c3.HycomFMRCFile.fetch(spec={'filter':f'archive=="{fmrc_archive_id}" && status == "downloaded" && subsetOptions.timeRange.start >= "{start}" && subsetOptions.timeRange.start <= "{end}"'})

# Step 2: Download files
print(t_0)
for file in files.objs:
    filename = os.path.basename(file.file.contentLocation)
    url = file.file.url
    # filesize = f'{file.file.contentLength / 1e6:.2f}MB'
    filesize = f'{file.file.contentLength} B'
    local_path = paths.DATA + '/temp/'
    start = time.time()
    if not os.path.exists(local_path + filename):
        c3.Client.copyFilesToLocalClient(url, '/tmp/')
        shutil.move('/tmp/' + filename, local_path)
    print(filename, filesize, f'{time.time() - start:.1f}s')


# %%

c3.HycomDataArchive.DownloadForecastFilesToLocal(
    HycomDataArchive=c3.HycomDataArchive.fetch(spec = { 'filter': 'dataset=="GOMu0.04/expt_90.1m000"' }).objs[0],
    t_0=t_0,
    x_0=x_0,
    x_T=x_T,
    n_days_ahead=n_days_ahead,
    local_folder='./data/hycom_forecast/'
)
# c3.HycomDataArchive.DownloadHindcastFilesToLocal(
#     HycomDataArchive=hycomDataArchive,
#     t_0=t_0,
#     x_0=x_0,
#     x_T=x_T,
#     n_days_ahead=n_days_ahead,
#     local_folder='./data/hycom_hindcast/'
# )