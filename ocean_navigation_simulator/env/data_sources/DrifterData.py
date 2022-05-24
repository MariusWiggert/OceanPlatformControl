'''
This class handles the inspection and loading of drifter data.

The drifter data is organized as follows:
 - index files: contain meta data about the .nc files and enables filtering of data
 - data files (netCDF): contain time series data for one drifter
'''

import datetime
import os
import webbrowser
from collections import namedtuple
from typing import Dict, List, Optional
from urllib.parse import urlparse

import folium
import ftputil
import pandas as pd
import xarray as xr
from folium import plugins
from shapely.geometry import Point, box


class DrifterData():
    def __init__(self, config: Dict, read_only=False):
        self.dataset = config['dataset']
        self.data_dir = config['data_dir']
        self.overlap_type = config['area_overlap_type']
        self.targeted_bbox = config['targeted_bbox']
        self.targeted_time_range = config['targeted_time_range']
        self.usr = config["usr"]
        self.pas = config["pas"]
        self.index_data: pd = None
        
        if not read_only:
            # download index files
            self.downloadIndexFiles(self.usr, self.pas)
        # load and join index files
        self.index_data = self.getIndexFileInfo()

    def downloadIndexFiles(self, usr: str, pas: str):
        #Provides the index files available on the ftp server

        if 'index_platform' in self.dataset.keys():
            indexes = self.dataset['index_files'] + [self.dataset['index_platform']]
        else:
            indexes = self.dataset['index_files']
        with ftputil.FTPHost(self.dataset['host'], usr, pas) as ftp_host:  # connect to CMEMS FTP
            for index in indexes:
                remotefile= "/".join(['Core', self.dataset['product'], self.dataset['name'], index])
                print('...Downloading ' + index)
                # localfile = os.path.join(os.getcwd(), 'data', 'drifter_data', 'index_files', index)
                localfile = os.path.join(self.data_dir, 'drifter_data', 'index_files', index)
                ftp_host.download(remotefile,localfile)  # remote, local

    def readIndexFileFromCWD(self, path2file: str, targeted_bbox: List[float], targeted_time_range: str, overlap_type: str="contains"):
        # Load as pandas dataframe the file in the provided path, additionally filter data during loading

        filename = os.path.basename(path2file)
        print('...Loading info from: '+filename)
        if targeted_bbox != None:
            raw_index_info =[]
            # skiprows needed due to header length of files
            chunks = pd.read_csv(path2file, skiprows=5,chunksize=1000)
            for chunk in chunks:
                chunk['spatialOverlap'] = chunk.apply(self.spatialOverlap, targeted_bbox=targeted_bbox, overlap_type=overlap_type, axis=1)
                chunk['temporalOverlap'] = chunk.apply(self.temporalOverlap, targeted_time_range=targeted_time_range,axis=1)
                raw_index_info.append(chunk[(chunk['spatialOverlap'] == True) & (chunk['temporalOverlap'] == True)])
            return pd.concat(raw_index_info)
        else:
            result = pd.read_csv(path2file, skiprows=5)
            try:
                result = result.rename(columns={"provider_edmo_code": "institution_edmo_code"})
            except Exception as e:
                pass
            return result

    def getIndexFileInfo(self):
        # Load and merge in a single entity all the information contained on each file descriptor of a given dataset
        # 1) Loading the index platform info as dataframe
        if 'index_platform' in self.dataset.keys():
            path2file = os.path.join(self.data_dir, 'drifter_data', 'index_files', self.dataset['index_platform'])
            indexPlatform = self.readIndexFileFromCWD(path2file, None, None)
            indexPlatform.rename(columns={indexPlatform.columns[0]: "platform_code" }, inplace = True)
            indexPlatform = indexPlatform.drop_duplicates(subset='platform_code', keep="first")
        # 2) Loading the index files info as dataframes
        netcdf_collections = []
        for filename in self.dataset['index_files']:
            path2file = os.path.join(self.data_dir,'drifter_data', 'index_files', filename)
            indexFile = self.readIndexFileFromCWD(path2file, self.targeted_bbox, self.targeted_time_range, overlap_type=self.overlap_type)
            netcdf_collections.append(indexFile)
        netcdf_collections = pd.concat(netcdf_collections)
        # 3) creating new columns: derived info
        netcdf_collections['netcdf'] = netcdf_collections['file_name'].str.split('/').str[-1]
        netcdf_collections['file_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[1]
        netcdf_collections['data_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[2]
        netcdf_collections['platform_code'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[3]
        # 4) Merging the information of all files
        if 'index_platform' in self.dataset.keys():
            headers = ['platform_code','wmo_platform_code', 'institution_edmo_code', 'last_latitude_observation', 'last_longitude_observation','last_date_observation']
            result = pd.merge(netcdf_collections,indexPlatform[headers],on='platform_code')
            return result
        else:
            return netcdf_collections

    def temporalOverlap(self, row, targeted_time_range: str):
        # Checks if a file contains data in the specified time range (targeted_time_range)
        try:
            date_format = "%Y-%m-%dT%H:%M:%SZ"
            targeted_ini = datetime.datetime.strptime(targeted_time_range.split('/')[0], date_format)
            targeted_end = datetime.datetime.strptime(targeted_time_range.split('/')[1], date_format)
            time_start = datetime.datetime.strptime(row['time_coverage_start'],date_format)
            time_end = datetime.datetime.strptime(row['time_coverage_end'],date_format)
            Range = namedtuple('Range', ['start', 'end'])
            r1 = Range(start=targeted_ini, end=targeted_end)
            r2 = Range(start=time_start, end=time_end)
            latest_start = max(r1.start, r2.start)
            earliest_end = min(r1.end, r2.end)
            delta = (earliest_end - latest_start).days + 1
            overlap = max(0, delta)
            if overlap != 0:
                result = True
            else:
                result = False
        except Exception as e:
            print("timeOverlap error")
        return result

    def spatialOverlap(self, row, targeted_bbox: List[str], overlap_type: str='contains'):
        # Checks if a file contains data in the specified area (targeted_bbox)

        result = False
        try:
            geospatial_lat_min = float(row['geospatial_lat_min'])
            geospatial_lat_max = float(row['geospatial_lat_max'])
            geospatial_lon_min = float(row['geospatial_lon_min'])
            geospatial_lon_max = float(row['geospatial_lon_max'])
            targeted_bounding_box = box(targeted_bbox[0], targeted_bbox[1], targeted_bbox[2], targeted_bbox[3])
            bounding_box = box(geospatial_lon_min, geospatial_lat_min,geospatial_lon_max, geospatial_lat_max)
            if overlap_type == "contains":
                if targeted_bounding_box.contains(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
                    result = True
                else:
                    result = False
            elif overlap_type == "intersects":
                if targeted_bounding_box.intersects(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
                    result = True
                else:
                    result = False

        except Exception as e:
            print("spatialOverlap error!\n", e)
        return result

    def downloadNCFile(self, file_link: str) -> str:
        '''
        receives file_link from index files and downloads the NC file
        '''
        file_link = "/".join(file_link.split("/")[3:])
        with ftputil.FTPHost(self.dataset["host"], self.usr, self.pas) as ftp_host:  # connect to CMEMS FTP
            remotefile = file_link
            # localfile = os.path.join(os.getcwd(), "data", "nc_files", remotefile.split("/")[-1])
            localfile = os.path.join(self.data_dir, "drifter_data", "nc_files", remotefile.split("/")[-1])
            if not os.path.exists(localfile):
                ftp_host.download(remotefile, localfile)
            else:
                return None

        return remotefile.split("/")[-1]

    @staticmethod
    def readNCFile(path2file: str) -> xr:
        ds = xr.open_dataset(path2file)
        ds.close()
        return ds

    def concatNCFiles(self, file_link_list: List[str]) -> xr.Dataset:
        if len(file_link_list) != 0:
            # download all files
            for link in file_link_list:
                self.downloadNCFile(file_link)
            # get list of paths to downloaded files
            base_dir = os.path.join(self.data_dir, "drifter_data", "nc_files")
            file_paths = [os.path.join(base_dir, file_name.split("/")[-1]) for file_name in file_link_list]
            # read files with xarray
            ds = xr.open_mfdataset(file_paths)

    @staticmethod
    def plotFoliumMapObject(foliumMap) -> None:
        output_file = "map.html"
        foliumMap.save(output_file)
        webbrowser.open_new(output_file)

    def plotXarray() -> None:
        pass
