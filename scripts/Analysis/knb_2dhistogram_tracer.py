#!/usr/bin/env python
# coding: utf-8

import argparse

parse_desc = """Find features, track, and plot all nexrad data in a given destination folder

The path is a string destination
Files in path must have a postfix of 'grid.nc'. 
threshold is the tracking threshold in dbz
speed is the tracking speed in tobac units. 
Site is a string NEXRAD location


Example
=======
python knb_2dhistogram_tracer.py --path="/Users/kelcy/DATA/20220604/" --site='KHGX' --dxy=0.5


"""


def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument('--path',metavar='path', required=True,dest='path',
                        action = 'store',help='path in which the data is located')
    parser.add_argument('-o', '--output_path',
                        metavar='filename template including path',
                        required=False, dest='outdir', action='store',
                        default='.', help='path in which the data is located')
    parser.add_argument('--site', metavar='site', required=True,
                        dest='site', action='store',
                        help='NEXRAD site code, e.g., khgx')
    parser.add_argument('--dxy', metavar='km', required=True,type=float,
                        dest='dxy', action='store',
                        help='Gridding x/y spacing in km')
#     parser.add_argument('--speed', metavar='value', required=True,type=float,
#                         dest='track_speed', action='store',
#                         help='Tracking speed, e.g., 1.0')
    return parser

# End parsing #

# Import libraries:

#2d Histogram code
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import os

# Disable a couple of warnings:
import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
warnings.filterwarnings("ignore", category=FutureWarning, append=True)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    
    track = xr.open_dataset(args.path+"Track_features_merges.nc")
    ts = pd.to_datetime(track['time'][100].values)
    date = ts.strftime('%Y%m%d')
    date2 = ts.strftime('%Y/%m/%d')
    print(date)


    maxrefl = []
    maxarea = []
    for num, i in enumerate(np.unique(track['track'].values)):
        if i <=0:
            continue
        j = np.where(track['feature_parent_track_id'].values == i)
        maxrefl.append(np.nanmax(track['max_reflectivity'].values[j]))
        maxarea.append(np.nanmax(track['feature_area'].values[j])*args.dxy*args.dxy)
    	
    	
    fig = plt.figure(figsize=(9,9))
    fig.set_canvas(plt.gcf().canvas)

    out = plt.hist2d(maxarea, maxrefl,range = [[0, 200.],[0, 75]])#,

    plt.colorbar()
    plt.xlabel('Maximum cell area, km$^2$', fontsize=14)
    plt.ylabel('Maximum reflectivity, dBz',fontsize=14)
    plt.title(date2 + ': Maximum cell reflectivity as a function of cell area',fontsize=16)
    fig.savefig(date+'_maxarea_maxrefl_2dhisto.png')
