from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from ocean_navigation_simulator.generative_error_model.utils import get_path_to_project

import datetime
import dateutil
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from collections import namedtuple
from typing import Dict, List
import ftputil
from ftputil.error import FTPIOError
import pandas as pd
import xarray as xr
from shapely.geometry import Point, box
import warnings
from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm

# ignore warnings that cannot be fixed for specific scenario of needing .loc and .iloc
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


@dataclass
class TargetedBbox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def get_bbox(self) -> List[float]:
        return [self.lon_min, self.lat_min, self.lon_max, self.lat_max]


@dataclass
class TargetedTimeRange:
    time_range: str

    def get_start(self) -> datetime.datetime:
        return dateutil.parser.isoparse(self.time_range.split("/")[0])

    def get_end(self) -> datetime.datetime:
        return dateutil.parser.isoparse(self.time_range.split("/")[1])


# TODO: directly obtain forecast/hindcast so one can call the interp func directly
class BuoyDataSource(ABC):
    """
    Abstract class that describes functionality of buoy data sources
    """

    def __init__(self, config: Dict, source: str):
        self.config = config
        self.buoy_config = self.config["buoy_config"][source]

        self.targeted_bbox = TargetedBbox(self.buoy_config["lon_range"][0],
                                          self.buoy_config["lon_range"][1],
                                          self.buoy_config["lat_range"][0],
                                          self.buoy_config["lat_range"][1]).get_bbox()
        self.targeted_time_range = TargetedTimeRange(self.buoy_config["time_range"])
        self.data_dir = os.path.join(get_path_to_project(os.getcwd()), self.config["data_dir"])
        self.data = self.get_buoy_data()
        self.index_data = None

    @abstractmethod
    def get_buoy_data(self) -> pd.DataFrame:
        """
        Returns the buoy data points as a pandas DataFrame
        """
        pass

    def interpolate_forecast(self, ocean_field: OceanCurrentField) -> pd.DataFrame:
        # TODO: need to ensure that buoy data and forecast/hindcast data overlap appropriately

        """
        Uses an OceanCurrentField object and interpolates the data to the
        spatio-temporal points of the buoy data.

        The ocean_field is a forecast.
        """

        self.data = self.interp_xarray(self.data, ocean_field, "forecast_data_source")
        print(f"Percentage of failed interp: {100*np.isnan(self.data['u_forecast']).sum()/self.data.shape[0]}%")
        return self.data

    def interpolate_hindcast(self, ocean_field: OceanCurrentField) -> pd.DataFrame:
        """
        Uses an OceanCurrentField object and interpolates the data to the
        spatio-temporal points of the buoy data.

        The ocean_field is a hindcast.
        """

        self.data = self.interp_xarray(self.data, ocean_field, "hindcast_data_source")
        print(f"Percentage of failed interp: {100*np.isnan(self.data['u_hindcast']).sum()/self.data.shape[0]}%")
        return self.data

    def interp_xarray(self, data: pd.DataFrame, ocean_field: OceanCurrentField, data_source: str, n: int = 10) -> pd.DataFrame:
        """
        Interpolates the hindcast data to spatio-temporal points of buoy measurements

        data_source: str {hindcast_data_source, forecast_data_source}
        """

        data[f"u_{data_source.split('_')[0]}"] = 0
        data[f"v_{data_source.split('_')[0]}"] = 0
        # convert time column to datetime
        data["time"] = pd.to_datetime(data["time"])

        for i in tqdm(range(0, data.shape[0], n)):
            interpolation = getattr(ocean_field, data_source).DataArray.interp(time=data["time"].iloc[i:i + n],
                                                                                 lon=data["lon"].iloc[i:i + n],
                                                                                 lat=data["lat"].iloc[i:i + n])
            # add columns to dataframe
            data[f"u_{data_source.split('_')[0]}"].iloc[i:i + n] = interpolation["water_u"].values.diagonal().diagonal()
            data[f"v_{data_source.split('_')[0]}"].iloc[i:i + n] = interpolation["water_v"].values.diagonal().diagonal()
        return data


class BuoyDataCopernicus(BuoyDataSource):
    def __init__(self, config: Dict, source="copernicus"):
        super().__init__(config, source)
        self.nc_links = None

    def get_buoy_data(self):
        """
        Main method which returns the data in the specified spatio-temporal range.
        It downloads the index files, combines index files into one xarray object,
        downloads NetCDF files if it contains data points in range, and finally
        the files are read and data is concatenated into a DataFrame.
        """
        # check if it should check for new buoy files
        download_index_files = self.buoy_config["dataset"]["download_index_files"]
        if download_index_files:
            self.download_index_files(self.buoy_config["usr"], self.buoy_config["pas"])
        self.index_data = self.get_index_file_info()

        self.nc_links = self.index_data["file_name"].tolist()
        self.download_all_NC_files(self.nc_links)

        file_list = [os.path.join(self.data_dir, "nc_files", file_link.split('/')[-1]) for file_link in self.nc_links]
        self.data = self.concat_buoy_data(file_list)

        return self.data

    def concat_buoy_data(self, file_list: List[str]) -> pd.DataFrame:
        """
        reads in all .nc files in the file_list
        filters data for desired time and space interval 
        creates a pandas.DataFrame
        """
        column_names = ["time", "lon", "lat", "u", "v", "buoy"]
        df = pd.DataFrame(columns=column_names)

        for file_path in file_list:
            ds, valid_file = self._read_NC_file(file_path)

            # catch corrupted files
            if valid_file is False:
                continue

            # select specific data
            time = ds["TIME"].values
            lon = ds["LONGITUDE"].values
            lat = ds["LATITUDE"].values
            u = ds["EWCT"].isel(DEPTH=-1).values  # problem here since deepest depth can be NaN
            v = ds["NSCT"].isel(DEPTH=-1).values
            file_name = [file_path.split("/")[-1].split(".")[0] for i in range(len(time))]
            buoy = [ds.attrs["platform_code"] for i in range(len(time))]
            df_temp = pd.DataFrame({"file_name": file_name, "buoy":buoy, "time":time, "lon":lon, "lat":lat, "u":u, "v":v})

            # change time column to datetime
            df["time"] = pd.to_datetime(df["time"])

            # filtering conditions
            lon_cond = ((df_temp["lon"] >= min(self.targeted_bbox[0], self.targeted_bbox[2])) & (df_temp["lon"] <= max(self.targeted_bbox[0], self.targeted_bbox[2])))
            lat_cond = ((df_temp["lat"] >= self.targeted_bbox[1]) & (df_temp["lat"] <= self.targeted_bbox[3]))
            time_cond = ((df_temp["time"] >= np.datetime64(self.targeted_time_range.get_start())) & (df_temp["time"] <= np.datetime64(self.targeted_time_range.get_end() + datetime.timedelta(hours=1))))

            # filtering and concat to df
            df_temp = df_temp.loc[(lon_cond & lat_cond & time_cond)]
            df = pd.concat([df, df_temp], ignore_index=True)

        # remove NaN rows -> created because of u and v at different depths.
        df = df.dropna()
        df = df.drop_duplicates(subset=["time", "buoy", "lon", "lat"], keep="first")
        return df

    def download_index_files(self, usr: str, pas: str):
        """
        The Copernicus buoy data includes meta data files called "index" files.
        This method downloads these index files
        """
        save_folder = os.path.join(self.data_dir, "index_files")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if "index_platform" in self.buoy_config["dataset"].keys():
            indexes = self.buoy_config["dataset"]["index_files"] + [self.buoy_config["dataset"]["index_platform"]]
        else:
            indexes = self.buoy_config["dataset"]["index_files"]
        with ftputil.FTPHost(self.buoy_config["dataset"]["host"], usr, pas) as ftp_host:  # connect to CMEMS FTP
            for index in indexes:
                remotefile = "/".join(['Core', self.buoy_config["dataset"]["product"], self.buoy_config["dataset"]["name"], index])
                print('...Downloading ' + index)
                localfile = os.path.join(save_folder, index)
                ftp_host.download(remotefile, localfile)  # remote, local

    def get_index_file_info(self) -> np.ndarray:
        """
        Load and merge index files in a single object all the information contained on each file descriptor of a given dataset
        """
        # 1) Loading the index platform info as dataframe
        if 'index_platform' in self.buoy_config["dataset"].keys():
            path2file = os.path.join(self.data_dir, "index_files", self.buoy_config["dataset"]["index_platform"])
            indexPlatform = self._read_index_file_from_CWD(path2file, None, None)
            indexPlatform.rename(columns={indexPlatform.columns[0]: "platform_code"}, inplace=True)
            indexPlatform = indexPlatform.drop_duplicates(subset='platform_code', keep="first")
        # 2) Loading the index files info as dataframes
        netcdf_collections = []
        for filename in self.buoy_config["dataset"]["index_files"]:
            path2file = os.path.join(self.data_dir, 'index_files', filename)
            index_file = self._read_index_file_from_CWD(path2file, self.targeted_bbox, self.targeted_time_range, overlap_type=self.buoy_config["area_overlap_type"])
            netcdf_collections.append(index_file)
        netcdf_collections = pd.concat(netcdf_collections)
        # 3) creating new columns: derived info
        netcdf_collections['netcdf'] = netcdf_collections['file_name'].str.split('/').str[-1]
        netcdf_collections['file_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[1]
        netcdf_collections['data_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[2]
        netcdf_collections['platform_code'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[3]
        # 4) Merging the information of all files
        if 'index_platform' in self.buoy_config["dataset"].keys():
            headers = ['platform_code', 'wmo_platform_code', 'institution_edmo_code', 'last_latitude_observation', 'last_longitude_observation','last_date_observation']
            result = pd.merge(netcdf_collections, indexPlatform[headers], on='platform_code')
            return result
        else:
            return netcdf_collections

    def _read_index_file_from_CWD(self, path2file: str, targeted_bbox: List[float], targeted_time_range: TargetedTimeRange, overlap_type: str="intersects"):
        # Load as pandas dataframe the file in the provided path, then filter data during loading

        filename = os.path.basename(path2file)
        print('...Loading info from: ' + filename)
        if targeted_bbox is not None:
            raw_index_info =[]
            # skiprows needed due to header length of files
            chunks = pd.read_csv(path2file, skiprows=5, chunksize=1000)
            for chunk in chunks:
                chunk['spatialOverlap'] = chunk.apply(self._spatial_overlap, targeted_bbox=targeted_bbox, overlap_type=overlap_type, axis=1)
                chunk['temporalOverlap'] = chunk.apply(self._temporal_overlap, targeted_time_range=targeted_time_range, axis=1)
                raw_index_info.append(chunk[(chunk['spatialOverlap'] == True) & (chunk['temporalOverlap'] == True)])
            return pd.concat(raw_index_info)
        else:
            result = pd.read_csv(path2file, skiprows=5)
            try:
                result = result.rename(columns={"provider_edmo_code": "institution_edmo_code"})
            except Exception as e:
                pass
            return result

    def _temporal_overlap(self, row, targeted_time_range: TargetedTimeRange):
        # Checks if a file contains data in the specified time range (targeted_time_range)
        try:
            targeted_ini = targeted_time_range.get_start()
            targeted_end = targeted_time_range.get_end()
            date_format = "%Y-%m-%dT%H:%M:%SZ"
            # targeted_ini = datetime.datetime.strptime(targeted_time_range.split('/')[0], date_format)
            # targeted_end = datetime.datetime.strptime(targeted_time_range.split('/')[1], date_format)
            # time_start = datetime.datetime.strptime(row['time_coverage_start'], date_format)
            # time_end = datetime.datetime.strptime(row['time_coverage_end'], date_format)
            time_start = dateutil.parser.isoparse(row["time_coverage_start"])
            time_end = dateutil.parser.isoparse(row["time_coverage_end"])
            Range = namedtuple('Range', ['start', 'end'])
            r1 = Range(start=targeted_ini, end=targeted_end)
            r2 = Range(start=time_start, end=time_end)
            latest_start = max(r1.start, r2.start)
            earliest_end = min(r1.end, r2.end)
            delta = (earliest_end - latest_start).days + 1
            overlap = max(0, delta)
            if overlap != 0:
                return True
            else:
                return False
        except Exception as e:
            print("timeOverlap error", e)

    def _spatial_overlap(self, row, targeted_bbox: List[str], overlap_type: str):
        # Checks if a file contains data in the specified area (targeted_bbox)

        def fix_negs(longitude):
            if longitude < 0:
                return longitude + 360
            else:
                return longitude

        result = False
        try:
            geospatial_lat_min = float(row['geospatial_lat_min'])
            geospatial_lat_max = float(row['geospatial_lat_max'])
            geospatial_lon_min = fix_negs(float(row['geospatial_lon_min']))
            geospatial_lon_max = fix_negs(float(row['geospatial_lon_max']))
            targeted_bounding_box = box(fix_negs(targeted_bbox[0]), targeted_bbox[1], fix_negs(targeted_bbox[2]), targeted_bbox[3])
            bounding_box = box(geospatial_lon_min, geospatial_lat_min, geospatial_lon_max, geospatial_lat_max)

            if overlap_type == "contains":
                if targeted_bounding_box.contains(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
                    result = True
            elif overlap_type == "intersects":
                if targeted_bounding_box.intersects(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
                    result = True

        except Exception as e:
            print("spatialOverlap error!\n", e)
        return result

    def _download_NC_file(self, file_link: str):
        """Receives file_link from index files and downloads the NC file.
        """
        remotefile = "/".join(file_link.split("/")[3:])
        save_folder = os.path.join(self.data_dir, "nc_files")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        localfile = os.path.join(save_folder, remotefile.split("/")[-1])
        if not os.path.isfile(localfile):
            with ftputil.FTPHost(self.buoy_config["dataset"]["host"], self.buoy_config["usr"], self.buoy_config["pas"]) as ftp_host:  # connect to CMEMS FTP
                print(f"downloading {localfile.split('/')[-1]}")
                try:
                    ftp_host.download(remotefile, localfile)
                except FTPIOError as obj:
                    print("File not accessible on remote host")
                    # remove from list, otherwise will try to read in later
                    self.nc_links.remove(file_link)
                    pass

    def download_all_NC_files(self, file_list: List[str]):
        """
        Takes list of .NC file links and downloads them
        """
        for file in file_list:
            self._download_NC_file(file)

    def _read_NC_file(self, path2file: str) -> xr:
        try:
            with xr.open_dataset(path2file, engine="netcdf4") as ds:
                return ds.load(), True
        except OSError as e:
            return None, False


class BuoyDataSofar(BuoyDataSource):

    def __init__(self, config: Dict, source="sofar"):
        super().__init__(config, source)

    def get_buoy_data(self) -> pd.DataFrame:
        """Main function which calls all sub-functions to retrieve Sofar buoy data
        in the specified spatio-temporal range."""

        self.index_data = self._read_index_file(os.path.join(self.config["data_dir"], "index.csv"),
                                                self.targeted_bbox,
                                                self.targeted_time_range)
        self.file_list = self.index_data["file_name"].tolist()
        data_dir = self.config["data_dir"]
        self.file_list = [os.path.join(data_dir, file) for file in self.file_list]
        self.data = self.concat_buoy_data(self.file_list)
        return self.data

    def concat_buoy_data(self, file_list: List[str]) -> pd.DataFrame:
        """
        reads in all .nc files in the file_list
        filters data for desired time and space interval
        creates a pandas.DataFrame
        """
        column_names = ["time", "lon", "lat", "u", "v", "buoy"]
        df = pd.DataFrame(columns=column_names)

        for file_path in file_list:
            df_temp = self._read_csv_file(file_path)
            df_temp = pd.DataFrame({"buoy": df_temp["buoy"],
                                    "time": df_temp["time"],
                                    "lon": df_temp["lon"],
                                    "lat": df_temp["lat"],
                                    "u": df_temp["u"],
                                    "v": df_temp["v"]})

            df_temp["time"] = pd.to_datetime(df_temp["time"])

            # filtering conditions
            lon_cond = ((df_temp["lon"] >= min(self.targeted_bbox[0], self.targeted_bbox[2])) & (df_temp["lon"] <= max(self.targeted_bbox[0], self.targeted_bbox[2])))
            lat_cond = ((df_temp["lat"] >= self.targeted_bbox[1]) & (df_temp["lat"] <= self.targeted_bbox[3]))
            time_cond = ((df_temp["time"] >= self.targeted_time_range.get_start()) & (df_temp["time"] <= self.targeted_time_range.get_end() + datetime.timedelta(hours=1)))

            # filtering and concat to df
            df_temp = df_temp.loc[(lon_cond & lat_cond & time_cond)]
            df = pd.concat([df, df_temp], ignore_index=True)

        # remove NaN rows -> created because of u and v at different depths.
        df = df.dropna()
        df = df.drop_duplicates(subset=["time", "buoy", "lon", "lat"], keep="first")
        return df

    def _read_index_file(self, path2file: str, targeted_bbox: List[float], targeted_time_range: TargetedTimeRange, overlap_type: str="intersects"):
        # Load as pandas dataframe the file in the provided path, then filter data during loading

        filename = os.path.basename(path2file)
        print('...Loading info from: ' + filename)
        if targeted_bbox is not None:
            raw_index_info = []
            chunks = pd.read_csv(path2file, chunksize=1000)
            for chunk in chunks:
                chunk['spatialOverlap'] = chunk.apply(self._spatial_overlap, targeted_bbox=targeted_bbox, overlap_type=overlap_type, axis=1)
                chunk['temporalOverlap'] = chunk.apply(self._temporal_overlap, targeted_time_range=targeted_time_range, axis=1)
                # print(f"Num of true spatial overlap files: {chunk['spatialOverlap'].sum()}.")
                # print(f"Num of true temporal overlap files: {chunk['temporalOverlap'].sum()}.")
                raw_index_info.append(chunk[(chunk['spatialOverlap'] == True) & (chunk['temporalOverlap'] == True)])
            return pd.concat(raw_index_info)
        else:
            raise ValueError("Please specify a Bounding Box")

    def _temporal_overlap(self, row, targeted_time_range: TargetedTimeRange):
        # Checks if a file contains data in the specified time range (targeted_time_range)
        try:
            targeted_ini = targeted_time_range.get_start()
            targeted_end = targeted_time_range.get_end()
            time_start = dateutil.parser.isoparse(row["time_coverage_start"])
            time_end = dateutil.parser.isoparse(row["time_coverage_end"])
            Range = namedtuple('Range', ['start', 'end'])
            r1 = Range(start=targeted_ini, end=targeted_end)
            r2 = Range(start=time_start, end=time_end)
            latest_start = max(r1.start, r2.start)
            earliest_end = min(r1.end, r2.end)
            delta = (earliest_end - latest_start).days + 1
            overlap = max(0, delta)
            if overlap != 0:
                return True
            else:
                return False
        except Exception as e:
            print("timeOverlap error", e)

    def _spatial_overlap(self, row, targeted_bbox: List[str], overlap_type: str):
        # Checks if a file contains data in the specified area (targeted_bbox)

        def fix_negs(longitude):
            if longitude < 0:
                return longitude + 360
            else:
                return longitude

        result = False
        try:
            geospatial_lat_min = float(row['geospatial_lat_min'])
            geospatial_lat_max = float(row['geospatial_lat_max'])
            geospatial_lon_min = fix_negs(float(row['geospatial_lon_min']))
            geospatial_lon_max = fix_negs(float(row['geospatial_lon_max']))
            targeted_bounding_box = box(fix_negs(targeted_bbox[0]), targeted_bbox[1], fix_negs(targeted_bbox[2]), targeted_bbox[3])
            bounding_box = box(geospatial_lon_min, geospatial_lat_min, geospatial_lon_max, geospatial_lat_max)

            if overlap_type == "contains":
                if targeted_bounding_box.contains(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
                    result = True
            elif overlap_type == "intersects":
                if targeted_bounding_box.intersects(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
                    result = True

        except Exception as e:
            print("spatialOverlap error!\n", e)
        return result

    def _read_csv_file(self, path2file: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path2file)
        except:
            raise IOError("Cannot read")


class BuoyDataNOOA(BuoyDataSource):
    # TODO: implement this using ftp (look at Copernicus class for this)

    def get_buoy_data(self, buoy_config: Dict) -> pd.DataFrame:
        return