from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import OceanCurrentField
from data_preprocessing import interp_hindcast_xarray

import datetime
import dateutil
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import yaml
from collections import namedtuple
from typing import Dict, List, Optional
from urllib.parse import urlparse
import ftputil
import pandas as pd
import xarray as xr
from shapely.geometry import Point, box


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


# TODO: Use DataField as inspiration here
class BuoyDataSource(ABC):
    """
    Abstract class that describes functionality of buoy data sources
    """

    def __init__(self, yaml_file_config: str, source: str):
        with open(yaml_file_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.data_config = config["data_config"][source]

        self.targeted_bbox = TargetedBbox(self.data_config["lon_range"][0],
                                     self.data_config["lon_range"][1],
                                     self.data_config["lat_range"][0],
                                     self.data_config["lat_range"][1]).get_bbox()
        self.targeted_time_range = TargetedTimeRange(self.data_config["time_range"])
        self.data = self.get_data(self.data_config)

    @abstractmethod
    def get_data(self, data_config: Dict) -> pd.DataFrame:
        """
        Returns the buoy data points as a pandas DataFrame
        """
        pass

    def interpolate_forecast(self, ocean_field: OceanCurrentField):
        """
        Uses an OceanCurrentField object and interpolates the data to the
        spatio-temporal points of the buoy data
        """

        self.data = interp_hindcast_xarray(self.data, ocean_field)
        return self.data


class BuoyDataCopernicus(BuoyDataSource):
    def __init__(self, config: Dict, source="copernicus"):
        super().__init__(config, source)

    def get_data(self, data_config: Dict):
        read_only = True
        if not read_only:
            # download index files
            self.download_index_files(data_config["usr"], data_config["pas"])
        # load and join index files
        self.index_data = self.get_index_file_info()

        # download files from index files
        nc_files = self.index_data["file_name"].tolist()
        self.download_all_NC_files(nc_files)

        # aggregate downloaded data
        file_list = [os.path.join(self.data_config["data_dir"], "drifter_data", "nc_files", file_link.split('/')[-1]) for file_link in nc_links]
        self.data = self.concat_buoy_data(file_list)

        return self.data

    def concat_buoy_data(self, file_list: List[str]) -> pd.DataFrame:
        """
        reads in all .nc files in the file_list
        filters data for desired time and space interval 
        creates a pandas.DataFrame
        """
        column_names = ["time", "lon", "lat", "u", "v", "buoy"]
        df = pd.DataFrame(columns = column_names)

        for file_path in file_list:
            ds = self._read_NC_file(file_path)

            # select specific data
            time = ds["TIME"].values
            lon = ds["LONGITUDE"].values
            lat = ds["LATITUDE"].values
            u = ds["NSCT"].isel(DEPTH=-1).values
            v = ds["EWCT"].isel(DEPTH=-1).values
            buoy = [file_path.split("/")[-1].split(".")[0] for i in range(len(time))]
            df_temp = pd.DataFrame({"time":time, "lon":lon, "lat":lat, "u":u, "v":v, "buoy":buoy})

            # change time column to datetime
            df["time"] = pd.to_datetime(df["time"])

            # filtering conditions TODO: fix issue with -ve values for lon
            lon_cond = ((df_temp["lon"] <= self.targeted_bbox[0]) & (df_temp["lon"] >= self.targeted_bbox[2]))
            lat_cond = ((df_temp["lat"] >= self.targeted_bbox[1]) & (df_temp["lat"] <= self.targeted_bbox[3]))
            time_cond = ((df_temp["time"] >= np.datetime64(self.targeted_time_range.get_start())) & (df_temp["time"] <= np.datetime64(self.targeted_time_range.get_end())))

            # filtering and concat to df
            df_temp = df_temp.loc[(lon_cond & lat_cond & time_cond)]
            df = pd.concat([df, df_temp])

        return df

    def download_index_files(self, usr: str, pas: str):
        """
        The Copernicus buoy data includes meta data files called "index" files.
        This method downloads these index files
        """

        if "index_platform" in self.data_config["dataset"].keys():
            indexes = self.data_config["dataset"]["index_files"] + [self.data_config["dataset"]["index_platform"]]
        else:
            indexes = self.data_config["dataset"]["index_files"]
        with ftputil.FTPHost(self.data_config["dataset"]["host"], usr, pas) as ftp_host:  # connect to CMEMS FTP
            for index in indexes:
                remotefile= "/".join(['Core', self.data_config["dataset"]["product"], self.data_config["dataset"]["name"], index])
                print('...Downloading ' + index)
                # localfile = os.path.join(os.getcwd(), 'data', 'drifter_data', 'index_files', index)
                localfile = os.path.join(self.data_config["data_dir"], "drifter_data", "index_files", index)
                ftp_host.download(remotefile, localfile)  # remote, local

    def get_index_file_info(self) -> np.ndarray:
        """
        Load and merge index files in a single object all the information contained on each file descriptor of a given dataset
        """
        # 1) Loading the index platform info as dataframe
        if 'index_platform' in self.data_config["dataset"].keys():
            path2file = os.path.join(self.data_config["data_dir"], "drifter_data", "index_files", self.data_config["dataset"]["index_platform"])
            indexPlatform = self._read_index_file_from_CWD(path2file, None, None)
            indexPlatform.rename(columns={indexPlatform.columns[0]: "platform_code" }, inplace = True)
            indexPlatform = indexPlatform.drop_duplicates(subset='platform_code', keep="first")
        # 2) Loading the index files info as dataframes
        netcdf_collections = []
        targeted_bbox = TargetedBbox(self.data_config["lon_range"][0],
                                     self.data_config["lon_range"][1],
                                     self.data_config["lat_range"][0],
                                     self.data_config["lat_range"][1]).get_bbox()
        targeted_time_range = TargetedTimeRange(self.data_config["time_range"])
        for filename in self.data_config["dataset"]["index_files"]:
            path2file = os.path.join(self.data_config["data_dir"],'drifter_data', 'index_files', filename)
            index_file = self._read_index_file_from_CWD(path2file, targeted_bbox, targeted_time_range, overlap_type=self.data_config["area_overlap_type"])
            netcdf_collections.append(index_file)
        netcdf_collections = pd.concat(netcdf_collections)
        # 3) creating new columns: derived info
        netcdf_collections['netcdf'] = netcdf_collections['file_name'].str.split('/').str[-1]
        netcdf_collections['file_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[1]
        netcdf_collections['data_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[2]
        netcdf_collections['platform_code'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[3]
        # 4) Merging the information of all files
        if 'index_platform' in self.data_config["dataset"].keys():
            headers = ['platform_code','wmo_platform_code', 'institution_edmo_code', 'last_latitude_observation', 'last_longitude_observation','last_date_observation']
            result = pd.merge(netcdf_collections,indexPlatform[headers],on='platform_code')
            return result
        else:
            return netcdf_collections

    def _read_index_file_from_CWD(self, path2file: str, targeted_bbox: List[float], targeted_time_range: TargetedTimeRange, overlap_type: str="contains"):
        # Load as pandas dataframe the file in the provided path, additionally filter data during loading

        filename = os.path.basename(path2file)
        print('...Loading info from: '+filename)
        if targeted_bbox != None:
            raw_index_info =[]
            # skiprows needed due to header length of files
            chunks = pd.read_csv(path2file, skiprows=5,chunksize=1000)
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

    def _spatial_overlap(self, row, targeted_bbox: List[str], overlap_type: str='intersects'):
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
        '''
        receives file_link from index files and downloads the NC file
        '''
        remotefile = "/".join(file_link.split("/")[3:])
        localfile = os.path.join(self.data_config["data_dir"], "drifter_data", "nc_files", remotefile.split("/")[-1])
        if not os.path.isfile(localfile):
            with ftputil.FTPHost(self.data_config["dataset"]["host"], self.data_config["usr"], self.data_config["pas"]) as ftp_host:  # connect to CMEMS FTP
                print(f"downloading {localfile.split('/')[-1]}")
                ftp_host.download(remotefile, localfile)

    def download_all_NC_files(self, file_list: List[str]):
        """
        Takes list of .NC file links and downloads them
        """
        for file in file_list:
            self._download_NC_file(file)

    def _read_NC_file(path2file: str) -> xr:
        with xr.open_dataset(path2file) as ds:
            return ds.load()


class BuoyDataSofar(BuoyDataSource):
    """
    Class to handle data from Sofar
    """
    def __init__(self, config: Dict):
        super().__init(config)

    def get_data(self, config: Dict):
        pass