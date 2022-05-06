from typing import Optional, Dict, List
import pandas as pd
import datetime
import os
from collections import namedtuple
import ftputil
from shapely.geometry import box, Point
from urllib.parse import urlparse
import webbrowser
import folium
from folium import plugins
import xarray as xr


class DrifterData():
    def __init__(self, dataset: Dict, config: Dict):
        self.dataset = dataset
        self.data_dir = config['data_dir']
        self.overlap_type = config['area_overlap_type']
        self.targeted_bbox = config['targeted_bbox']
        self.targeted_time_range = config['targeted_time_range']
        self.usr = config["usr"]
        self.pas = config["pas"]
        self.index_data: pd= None

        # download index files
        self.downloadIndexFiles(self.usr, self.pas)
        # aggreagate and load index files
        self.index_data = self.getIndexFileInfo()

    def downloadIndexFiles(self, usr: str, pas: str):
        #Provides the index files available at the ftp server
        '''
        TODO: better file handling e.g. specify dir to save to.
        '''
        indexes = self.dataset['index_files'] + [self.dataset['index_platform']]
        with ftputil.FTPHost(self.dataset['host'], usr, pas) as ftp_host:  # connect to CMEMS FTP
            for index in indexes:
                remotefile= "/".join(['Core', self.dataset['product'], self.dataset['name'], index])
                print('...Downloading ' + index)
                # localfile = os.path.join(os.getcwd(), 'data', 'drifter_data', 'index_files', index)
                localfile = os.path.join(self.data_dir, 'drifter_data', 'index_files', index)
                ftp_host.download(remotefile,localfile)  # remote, local

    def readIndexFileFromCWD(self, path2file: str, targeted_bbox: List[float], overlap_type: str="contains"):
        #Load as pandas dataframe the file in the provided path
        '''
        TODO: instead of targeted_bbox add dict of structure {func: data}
        example: filter_dict = {spatialOverlap: targeted_bbox, temporalOverlap: targeted_time_range}
        loop over keys of dict to apply multiple functions and 
        '''
        filename = os.path.basename(path2file)
        print('...Loading info from: '+filename)
        if targeted_bbox != None:
            raw_index_info =[]
            # skiprows needed due to header length of files
            chunks = pd.read_csv(path2file, skiprows=5,chunksize=1000)
            for chunk in chunks:
                chunk['spatialOverlap'] = chunk.apply(self.spatialOverlap, targeted_bbox=targeted_bbox, overlap_type=overlap_type, axis=1)
                raw_index_info.append(chunk[chunk['spatialOverlap'] == True])
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
        path2file = os.path.join(self.data_dir, 'drifter_data', 'index_files', self.dataset['index_platform'])
        indexPlatform = self.readIndexFileFromCWD(path2file, None)
        indexPlatform.rename(columns={indexPlatform.columns[0]: "platform_code" }, inplace = True)
        indexPlatform = indexPlatform.drop_duplicates(subset='platform_code', keep="first")
        # 2) Loading the index files info as dataframes
        netcdf_collections = []
        for filename in self.dataset['index_files']:
            path2file = os.path.join(self.data_dir,'drifter_data', 'index_files', filename)
            indexFile = self.readIndexFileFromCWD(path2file, self.targeted_bbox, overlap_type=self.overlap_type)
            netcdf_collections.append(indexFile)
        netcdf_collections = pd.concat(netcdf_collections)
        # 3) creating new columns: derived info
        netcdf_collections['netcdf'] = netcdf_collections['file_name'].str.split('/').str[-1]
        netcdf_collections['file_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[1]
        netcdf_collections['data_type'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[2]
        netcdf_collections['platform_code'] = netcdf_collections['netcdf'].str.split('.').str[0].str.split('_').str[3]
        # 4) Merging the information of all files
        headers = ['platform_code','wmo_platform_code', 'institution_edmo_code', 'last_latitude_observation', 'last_longitude_observation','last_date_observation']
        result = pd.merge(netcdf_collections,indexPlatform[headers],on='platform_code')
        return result

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
        '''
        TODO: could replace intersects function with contains function
        '''
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

    def downloadNCFile(self, file_link: str) -> None:
        '''
        receives file_link from index files and downloads the NC file
        '''
        file_link = "/".join(file_link.split("/")[3:])
        with ftputil.FTPHost(self.dataset["host"], self.usr, self.pas) as ftp_host:  # connect to CMEMS FTP
            remotefile = file_link
            # localfile = os.path.join(os.getcwd(), "data", "nc_files", remotefile.split("/")[-1])
            localfile = os.path.join(self.data_dir, "drifter_data", "nc_files", remotefile.split("/")[-1])
            ftp_host.download(remotefile, localfile)

    @staticmethod
    def readNCFile(path2file: str) -> xr:
        ds = xr.open_dataset(path2file)
        ds.close()
        return ds

    def aggregateNCFiles(self, path2file_list: List[str]) -> xr:
        pass

    @staticmethod
    def plotFoliumMapObject(foliumMap):
        output_file = "map.html"
        foliumMap.save(output_file)
        webbrowser.open_new(output_file)