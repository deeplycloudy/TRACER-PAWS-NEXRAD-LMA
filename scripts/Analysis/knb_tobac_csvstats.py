#!/usr/bin/env python
# coding: utf-8

import argparse

parse_desc = """Find features, track, and plot all nexrad data in a given destination folder

The path is a string destination
Files in path must have a postfix of '.nc', this will be searched for internally.
tobacpath: path to the tobac feature, track etc. netcdf files.
celltrack: level at which to evaluate the statistics - cell or track level.


Example
=======
python knb_tobac_csvstats.py --tobacpath="/Users/kelcy/PYTHON/tracer-jcss_EBfork/tobac_Save_20220322/" --celltrack="cell"


"""




def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(
        "--tobacpath",
        metavar="tobacpath",
        required=True,
        dest="tobacpath",
        action="store",
        help="path in which the tobac tracked files are located",
    )
    parser.add_argument(
        "--celltrack",
        metavar="celltrack",
        required=True,
        dest="celltrack",
        action="store",
        help="Stats are given at the cell or the track level specified here. Default is Track",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        metavar="filename template including path",
        required=False,
        dest="outdir",
        action="store",
        default=".",
        help="path in which the data is located",
    )
    return parser

import numpy as np
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import csv
from glob import glob
from pathlib import Path
from pandas.core.common import flatten

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()



    file = glob(args.tobacpath + "timeseries*.nc")
    track = xr.open_dataset(file[0])
#  /Users/kelcy/PYTHON/tracer-jcss_EBfork/tobac_Save_20160331/timeseries_data_20160331_new.nc
    data = xr.open_dataset(args.tobacpath+"Track_features_merges.nc")
    #Assumes the tobac path name has the date at the end.
    date = args.tobacpath[-9:-1]
    ltg_onlydir = 0
    kdp_zdr_ltgdir = 0
    kdp_ltgdir = 0
    zdr_ltgdir = 0
    kdp_zdrdir = 0
    kdpdir = 0
    zdrdir = 0
    nothing = 0
    id_num = []
    for num,i in enumerate(track[args.celltrack].values):
        if i == -1:
            continue
        if args.celltrack == "cell":
            find = np.where(data['feature_parent_track_id'] == i)
            j = np.where(data['feature_parent_cell_id'].values == i)
        else:
            find = np.where(data['feature_parent_track_id'] == i)
            j = np.where(data['feature_parent_track_id'].values == i)
#Use only cells/Tracks within 150km of the radar
        locsx = data['feature_projection_x_coordinate'].values[find]
        locsy = data['feature_projection_y_coordinate'].values[find]
        if (np.mean(np.abs(locsx)) > 150000.) and (np.mean(np.abs(locsy)) > 150000.):
            continue
        
        #ZDR ONLY
        if any(val > 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            zdrdir += 1

        #KDP ONLY
        if all(val <= 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            kdpdir += 1
        
        #LIGHTNING ONLY   
        if all(val <= 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            ltg_onlydir += 1
        
        #LIGHTNING WITH KDP AND ZDR
        if any(val > 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            kdp_zdr_ltgdir += 1 

		#KDP WITH LIGHTNING
        if all(val <= 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            kdp_ltgdir += 1
        
        #ZDR WITH LIGHTNING
        if any(val > 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            zdr_ltgdir += 1
            
        #KDP WITH ZDR, NO LIGHTNING
        if any(val > 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and any(val > 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            kdp_zdrdir += 1
        
        #NO KDP, ZDR, OR LIGHTNING
        if all(val <= 0 for val in list(flatten(track['feature_zdrvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_kdpvol'].values[j]))) and all(val <= 0 for val in list(flatten(track['feature_flash_count'].values[j]))):
            nothing += 1

    # Define the structure of the data
    header = ["nothing", "zdrdir","kdpdir","kdp_zdrdir","ltg_onlydir","kdp_zdr_ltgdir","kdp_ltgdir","zdr_ltgdir"]


    results = [nothing,zdrdir,kdpdir,kdp_zdrdir,ltg_onlydir,kdp_zdr_ltgdir,kdp_ltgdir,zdr_ltgdir]
    print(results)
    print(sum(results))
# # 	1. Open a new CSV file
    with open(args.tobacpath+str(date)+'_results.csv', 'w') as file:
#     # 2. Create a CSV writer
        writer = csv.writer(file)
#     # 3. Write data to the file
        writer.writerow(header)
        writer.writerow(results)
 


