# source: "https://github.com/MariusWiggert/OceanPlatformControl/blob/Jerome_reinforcement_learning/
# ocean_navigation_simulator/environment/ArenaFactory.py"

import datetime
import os
from typing import List

from c3python import C3Python


def get_user_dir():
    path_parts = os.getcwd().split("/")
    return "/".join(path_parts[:3])


def get_path_to_project(static_path: str) -> str:
    file_split = static_path.split("/")
    end_idx = file_split.index("OceanPlatformControl")
    relative_path = "/".join(file_split[: end_idx + 1])
    return relative_path


class C3Downloader:
    def __init__(self):
        print(os.path.join(get_user_dir(), ".ssh/c3-rsa"))
        c3 = C3Python(
            url="https://dev01-seaweed-control.c3dti.ai",
            tenant="seaweed-control",
            tag="dev01",
            keyfile=os.path.join(get_user_dir(), ".ssh/c3-rsa"),
            username="killian.kaempf@berkeley.edu",
        )
        self.c3 = c3.get_c3()
        self.data_dir_fc = os.path.join(get_path_to_project(os.getcwd()), "data_ablation_study/fc")
        self.data_dir_hc = os.path.join(get_path_to_project(os.getcwd()), "data_ablation_study/hc")

    def get_files_list(
        self,
        source: str,
        type_of_data: str,
        region: str,
        time_interval: List[datetime.datetime],
    ):
        """
        Args:
            source: str {Copernicus, Hycom}
            type_of_data: str {forecast, hindcast}
            region: str {Region 1, Region 2, etc., GoM}
            time_interval: List[datetime.datetime}
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

        def conversion(x):
            return x or ""

        try:
            region_name = [name for name in names if region in conversion(name)]
            idx = names.index(region_name[0])
        except Exception:
            print(f"regions available: {names}")
            raise ValueError(
                f"Specified region name '{region}' is not a {source} {type_of_data.capitalize()} Data Archive!"
            )
        type_map = {"forecast": "fmrcArchive", "hindcast": "hindcastArchive"}
        if type_of_data not in list(type_map.keys()):
            raise ValueError("Type of data invalid choose from [forecast, hindcast].")
        specific_archive_id = getattr(data_archive.objs[idx], type_map[type_of_data]).id
        print(f"Archive ID: {specific_archive_id}")

        # Step 2: get relevant files in region within time range
        type_map = {"forecast": "FMRC", "hindcast": "Hindcast"}
        file_object_name = f"{source}{type_map[type_of_data]}File"
        files_list = getattr(self.c3, file_object_name).fetch(
            spec={
                "filter": f"archive=='{specific_archive_id}' && status=='downloaded' && {time_filter}",
                "order": "ascending(subsetOptions.timeRange.start)",
            }
        )
        return files_list.objs

    def download_files(self, files_list: List[C3Python], download_folder: str, is_forecast):
        if files_list is None:
            raise ValueError("No files present on C3 with specified requirements!")
        download_root = os.path.join(
            self.data_dir_fc if is_forecast else self.data_dir_hc, download_folder
        )
        print(f"Downloading {'forecasts' if is_forecast else 'hindcasts'} to: {download_root}.\n")
        for file in files_list:
            filename = os.path.basename(file.file.contentLocation)
            url = file.file.url
            filesize = file.file.contentLength
            if (
                not os.path.exists(os.path.join(download_root, filename))
                or os.path.getsize(os.path.join(download_root, filename)) != filesize
            ):
                self.c3.Client.copyFilesToLocalClient(url, download_root)
                print(f"Downloaded {filename}")
                # TODO: check file size!
            if os.path.getsize(os.path.join(download_root, filename)) != filesize:
                raise Exception(
                    f"Downloaded forecast file with incorrect file size. Should be {filesize}B but is {os.path.getsize(download_root + filename)}B."
                )
            else:
                os.system(f"touch {os.path.join(download_root, filename)}")


if __name__ == "__main__":
    c3_downloader = C3Downloader()
    april_time_interval = [
        datetime.datetime(2022, 3, 28, 12, 0, 0),
        datetime.datetime(2022, 4, 29, 12, 0, 0),
    ]
    files = c3_downloader.get_files_list("Hycom", "hindcast", "GoM", april_time_interval)
    c3_downloader.download_files(files, "april", False)
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", april_time_interval)
    c3_downloader.download_files(files, "april", True)

    may_time_interval = [
        datetime.datetime(2022, 5, 1, 0, 0, 0),
        datetime.datetime(2022, 6, 1, 0, 0, 0),
    ]
    files = c3_downloader.get_files_list("Hycom", "hindcast", "GoM", may_time_interval)
    c3_downloader.download_files(files, "may", False)
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", may_time_interval)
    c3_downloader.download_files(files, "may", True)

    june_time_interval = [
        datetime.datetime(2022, 6, 1, 0, 0, 0),
        datetime.datetime(2022, 7, 1, 0, 0, 0),
    ]
    files = c3_downloader.get_files_list("Hycom", "hindcast", "GoM", june_time_interval)
    c3_downloader.download_files(files, "june", False)
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", june_time_interval)
    c3_downloader.download_files(files, "june", True)

    m = "july"
    time_interval = [datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 8, 1, 0, 0, 0)]
    files = c3_downloader.get_files_list("Hycom", "hindcast", "GoM", time_interval)
    c3_downloader.download_files(files, m, False)
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", time_interval)
    c3_downloader.download_files(files, m, True)

    m = "august"
    time_interval = [datetime.datetime(2022, 8, 1, 0, 0, 0), datetime.datetime(2022, 9, 1, 0, 0, 0)]
    files = c3_downloader.get_files_list("Hycom", "hindcast", "GoM", time_interval)
    c3_downloader.download_files(files, m, False)
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", time_interval)
    c3_downloader.download_files(files, m, True)

    m = "september"
    time_interval = [
        datetime.datetime(2022, 9, 1, 0, 0, 0),
        datetime.datetime(2022, 9, 16, 0, 0, 0),
    ]
    files = c3_downloader.get_files_list("Hycom", "hindcast", "GoM", time_interval)
    c3_downloader.download_files(files, m, False)
    files = c3_downloader.get_files_list("Copernicus", "forecast", "GoM", time_interval)
    c3_downloader.download_files(files, m, True)
    print("--- over ---")
