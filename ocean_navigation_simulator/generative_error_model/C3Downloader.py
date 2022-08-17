# source: "https://github.com/MariusWiggert/OceanPlatformControl/blob/Jerome_reinforcement_learning/
# ocean_navigation_simulator/environment/ArenaFactory.py"

from c3python import C3Python
import datetime
from typing import Tuple, List
import os


class C3Downloader:
    def __init__(self):
        c3 = C3Python(
            url='https://dev01-seaweed-control.c3dti.ai',
            tenant='seaweed-control',
            tag='dev01',
            keyfile='/home/jonas/.ssh/c3-rsa',
            username='jonas.dieker@berkeley.edu',
        ).get_c3()
        self.c3 = c3

    def get_files_list(self, source: str, type_of_data: str, region: str, time_interval: List[datetime.datetime]):
        time_start = time_interval[0]
        time_end = time_interval[1]
        time_filter = f"subsetOptions.timeRange.start >= '{time_start}' && subsetOptions.timeRange.start <= '{time_end}'"
        source = source.capitalize()

        # Step 1: Get id of specified region
        archive_type = f"{source}DataArchive"
        data_archive = getattr(self.c3, archive_type).fetch()
        names = []
        for i in range(data_archive.count):
            names.append(data_archive.objs[i].name)
        conversion = lambda x: x or ""
        try:
            region_name = [name for name in names if region in conversion(name)]
            idx = names.index(region_name[0])
        except:
            raise ValueError(f"Specified region name '{region}' is not a {source} {type_of_data.capitalize()} Data Archive!")
        type_map = {"forecast": "fmrcArchive", "hincast": "hindcastArchive"}
        if type_of_data not in list(type_map.keys()):
            raise ValueError("Type of data invalid choose from [forecast, hindcast].")
        specific_archive_id = getattr(data_archive.objs[idx], type_map[type_of_data]).id

        # Step 2: get relevant files in region within time range
        type_map = {"forecast": "FMRC", "hindcast": "Hindcast"}
        file_object_name = f"{source}{type_map[type_of_data]}File"
        files_list = getattr(self.c3, file_object_name).fetch(
            spec={"filter": f"archive=='{specific_archive_id}' && status=='downloaded' && {time_filter}",
                  "order": "ascending(subsetOptions.timeRange.start)"})
        return files_list.objs

    def download_files(self, files_list: List[C3Python], download_folder: str):
        for file in files_list:
            filename = os.path.basename(file.file.contentLocation)
            url = file.file.url
            filesize = file.file.contentLength
            if not os.path.exists(os.path.join(download_folder, filename)) or os.path.getsize(os.path.join(download_folder, filename)) != filesize:
                self.c3.Client.copyFilesToLocalClient(url, download_folder)
                print(f"Downloaded {filename}")
                # TODO: check file size!
            if os.path.getsize(os.path.join(download_folder, filename)) != filesize:
                raise Exception(
                    f"Downloaded forecast file with incorrect file size. Should be {filesize}B but is {os.path.getsize(download_folder + filename)}B.")
            else:
                os.system(f"touch {os.path.join(download_folder, filename)}")


if __name__ == "__main__":
    c3_downloader = C3Downloader()
    time_interval = [datetime.datetime(2022, 5, 10, 12, 0, 0), datetime.datetime(2022, 5, 25, 12, 0, 0)]
    files = c3_downloader.get_files_list("Copernicus", "forecast", "Region 4", time_interval)
    c3_downloader.download_files(files, "/home/jonas/Downloads/script")
