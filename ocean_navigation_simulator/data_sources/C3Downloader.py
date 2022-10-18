from ocean_navigation_simulator.utils.misc import get_c3

from c3python import C3Python
import datetime
from typing import List, Optional
import os

## How to get c3 Keyfile set up
# Step 1: generate the public and private keys locally on your computer
# in terminal run 'openssl genrsa -out c3-rsa.pem 2048' -> this generates the private key in the c3-rsa.pem file
# for public key from it run 'openssl rsa -in c3-rsa.pem -outform PEM -pubout -out public.pem'
# Step 2: move the c3-rsa.pem file to a specific folder
# Step 3: Log into C3, start jupyter service and in a cell update your users public key by
# user = c3.User.get("mariuswiggert@berkeley.edu")
# user = user.get("publicKey")
# user.publicKey = "<public key from file>"
# user.merge()

KEYFILE = 'setup/keys/c3-rsa-marius.pem'
USERNAME = 'mariuswiggert@berkeley.edu'


class C3Downloader:
    """Downloads forecast and hindcast files from C3.
    [Please check above for generating a keyfile associated with your Berkeley email.]
    """

    def __init__(self):
        c3 = get_c3()
        self.c3 = c3

    def get_files_list(self, source: str, type_of_data: str, region: str, time_interval: List[datetime.datetime]):
        """
        Args:
            source: str {Copernicus, Hycom}
            type_of_data: str {forecast, hindcast}
            region: str {Region 1, Region 2, etc., GoM}
            time_interval: List[datetime.datetime]
        """
        time_start = time_interval[0]
        time_end = time_interval[1]
        time_filter = f"subsetOptions.timeRange.start >= '{time_start}' && subsetOptions.timeRange.start <= '{time_end}'"
        source = source.capitalize()

        # Step 1: Get id of specified region
        archive_type = f"{source}DataArchive"
        data_archive = getattr(self.c3, archive_type).fetch()
        names = []
        for i in range(data_archive.count):
            name = data_archive.objs[i].name
            if data_archive.objs[i].name is None:
                name = "GoM"
            names.append(name)
        conversion = lambda x: x or ""
        try:
            region_name = [name for name in names if region in conversion(name)]
            idx = names.index(region_name[0])
        except:
            raise ValueError(f"Specified region name '{region}' is not a {source} {type_of_data.capitalize()} Data Archive!")
        type_map = {"forecast": "fmrcArchive", "hindcast": "hindcastArchive"}
        if type_of_data not in list(type_map.keys()):
            raise ValueError("Type of data invalid choose from [forecast, hindcast].")
        specific_archive_id = getattr(data_archive.objs[idx], type_map[type_of_data]).id
        print(f"Archive ID: {specific_archive_id}")

        # Step 2: get relevant files in region within time range
        type_map = {"forecast": "FMRC", "hindcast": "Hindcast"}
        file_object_name = f"{source}{type_map[type_of_data]}File"
        files_list = getattr(self.c3, file_object_name).fetch(
            spec={"filter": f"archive=='{specific_archive_id}' && status=='downloaded' && {time_filter}",
                  "order": "ascending(subsetOptions.timeRange.start)"})
        return files_list.objs

    def download_files(self, files_list: List[C3Python], download_folder: str):
        if files_list is None:
            raise ValueError("No files present on C3 with specified requirements!")
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        print(f"Downloading files to: {download_folder}.\n")
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


def test():
    """Adjust this function for manual downloading or testing.
    """
    c3_downloader = C3Downloader()
    time_interval = [datetime.datetime(2022, 4, 21, 12, 0, 0), datetime.datetime(2022, 4, 22, 12, 0, 0)]
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", time_interval)
    c3_downloader.download_files(files, "/dir/where/to/save/files")


if __name__ == "__main__":
    test()
