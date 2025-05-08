#!/usr/bin/env python
# coding: utf-8

# MODIFIED 
# 20230502: KNB
# 20230630: ECB
#20231206: KNB

import argparse


parse_desc = """Find features, track, and plot all nexrad data in a given destination folder

The path is a string destination
Files in path must have a postfix of '.nc', this will be searched for internally.
Three paths are required: 
path: path to the radar data, currently only NEXRAD data is supported in this version
lmapath: path to the lma flash sorted data
tobacpath: path to the tobac feature, track etc. netcdf files.
type: Name of the type of data (NEXRAD/POLARRIS/NUWRF) given as all uppercase string. Currently only NEXRAD is supported.


Example
=======
python knb_tobac_timseriesNUWRF.py --nuwrfpath="/Users/kelcy/DATA/PERiLS LMA deployments/V1.1_data/gridded/20220322_KGWX/" --tobacpath="/Users/kelcy/PYTHON/tracer-jcss_EBfork/tobac_Save_20220322/" --type="POLARRIS"


Variables
=========
About the variables created and/or used by this script:
feature_grid_cell_count - from tobac, maybe not reliable.
feature_area - count of grid boxes in the 2D feature footprint.
feature_maxrefl -  max reflectivity anywhere in 3D in feature.
feature_zdrvol - count of grid boxes Zdr above 1 dB, in 3 km slab above melting level in feature. Thresholds from van Lier Walqui.

"""

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(
        "--nuwrfpath",
        metavar="nuwrfpath",
        required=True,
        dest="nuwrfpath",
        action="store",
        help="path in which the flash sorted lma data is located",
    )
    parser.add_argument(
        "--tobacpath",
        metavar="tobacpath",
        required=True,
        dest="tobacpath",
        action="store",
        help="path in which the tobac tracked files are located",
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
    parser.add_argument(
        "--type",
        metavar="data type",
        required=False,
        dest="data_type",
        action="store",
        help="Datat name type, e.g., NEXRAD, POLARRIS, NUWRF",
    )
    return parser


# End parsing #

# Import packages

import numpy as np
import warnings
import xarray as xr
import random
from scipy import ndimage
from datetime import datetime
import cartopy.crs as ccrs
try:
    import pyproj

    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False


# VERSION FOR EACH FEATURE AND CELL IN XARRAY DATASET


import gc
import tracemalloc
from matplotlib import cm

from glob import glob
import pandas as pd
from netCDF4 import Dataset

import pyart
from scipy import ndimage as ndi
from scipy import spatial
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from geopy.distance import geodesic, great_circle
import os
from scipy.interpolate import griddata
from pandas.core.common import flatten
import warnings
from skimage.morphology import disk, dilation
from pyxlma.lmalib.traversal import OneToManyTraversal
import xwrf
warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
warnings.filterwarnings("ignore", category=FutureWarning, append=True)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
import time



def main(args):
    ###LOAD DATA

    
    files = sorted(glob(args.nuwrfpath + "wrf2dout_d02_*"))
    time_arr = []
    for i in files:
        time_arr.append(pd.to_datetime(i[-19:],format = "%Y-%m-%d_%H:%M:%S"))
    time_arr = np.array(time_arr)
    nuwrf_df = pd.DataFrame(time_arr, columns=['nuwrf_time'])

    data1 = xr.open_dataset(files[0],decode_times=False)
    drop_list = list(np.sort(list(data1.variables)))
    good_list = list(flatten(list(('COMDBZ', 'Times','XLAT','XLONG','XTIME','FRZLVL','TPW','LWP','IWP',
                             'CLWP','RWP', 'CIWP','SIWP','GIWP','HIWP','LIGHTDIS','LIGHTDENS',
                              ))))# + (args.add_var,))))
    drop_list = [e for e in drop_list if e not in good_list]



    
    # Read in feature information
    savedir = args.tobacpath
    xrdata = xr.open_dataset(savedir + "Track_features_merges.nc")
    
    # Remove all tracks that don't reach 35 dBZ (or whatever thresh we set below)
    track_dBZ_thresh = 30.0
    feature_is_bright = (xrdata['max_reflectivity'] > track_dBZ_thresh)
    reduced_track_ids = np.unique(xrdata[{'feature':feature_is_bright}].feature_parent_track_id)
    traversal = OneToManyTraversal(xrdata, ('track','cell','feature'), ('cell_parent_track_id',   'feature_parent_cell_id'))
    xrdata = traversal.reduce_to_entities('track', reduced_track_ids)
    print('reduced')
    
    dxy = data1.DX / 1000.0
    dt = 5.0 #data1.DT
    print("Mean spacing of data in x:", dxy)

    val = pd.to_datetime(xrdata["time"].values)

    feature_SRC = dict()
    feature_FED = dict()
    feature_TPW_max = dict()
    feature_LWP_max = dict()
    feature_IWP_max = dict()
    feature_CLWP_max = dict()
    feature_RWP_max = dict()
    feature_CIWP_max = dict()
    feature_SIWP_max = dict()
    feature_GIWP_max = dict()
    feature_HIWP_max = dict()
    feature_TPW_mean = dict()
    feature_LWP_mean = dict()
    feature_IWP_mean = dict()
    feature_CLWP_mean = dict()
    feature_RWP_mean = dict()
    feature_CIWP_mean = dict()
    feature_SIWP_mean = dict()
    feature_GIWP_mean = dict()
    feature_HIWP_mean = dict()
    feature_TPW_mean_all = dict()
    feature_LWP_mean_all = dict()
    feature_IWP_mean_all = dict()
    feature_CLWP_mean_all = dict()
    feature_RWP_mean_all = dict()
    feature_CIWP_mean_all = dict()
    feature_SIWP_mean_all = dict()
    feature_GIWP_mean_all = dict()
    feature_HIWP_mean_all = dict()

    rhgt = (np.array(
            [
                0,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
                14.5,
                15,
            ]
        )
        * 1000.0)


#     # del pdata
    start = time.time()
     
    print(np.nanmax(np.unique(xrdata["feature_time_index"].values)))
    print(np.nanmax(xrdata["feature"].values))
    
    
    for num, i in enumerate(np.unique(xrdata["feature_time_index"].values)):
        print(i)

        ids = np.where(xrdata["feature_time_index"].values == i)
        ids = xrdata["feature"].values[ids]
        features_this_frame = xrdata["segmentation_mask"].data[i, :, :]
    
        t1 = pd.to_datetime(xrdata['time'][i].values)
        nuwrf_file = np.argmin(abs(nuwrf_df - t1))

        nudata = xr.open_dataset(files[nuwrf_file], engine="netcdf4",decode_times=False,
            drop_variables=drop_list).xwrf.postprocess()
        nudata = nudata.rename_dims({'Time': 'time'})
        nudata['time'] = nudata['Time']
        print(nudata['time'].values)
    # Find the index in rhgt that correponds to the melting level given in the command line
       
        for nid, f in enumerate(ids):
            print(f)
            this_feature = (features_this_frame == f)

    
            feature_TPW_max[f] = nudata['TPW'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_TPW_mean[f] = nudata['TPW'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_TPW_mean_all[f] = nudata['TPW'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_LWP_max[f] = nudata['LWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_LWP_mean[f] = nudata['LWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_LWP_mean_all[f] = nudata['LWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_IWP_max[f] = nudata['IWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_IWP_mean[f] = nudata['IWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_IWP_mean_all[f] = nudata['IWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_CLWP_max[f] = nudata['CLWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_CLWP_mean[f] = nudata['CLWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_CLWP_mean_all[f] = nudata['CLWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_RWP_max[f] = nudata['RWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_RWP_mean[f] = nudata['RWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_RWP_mean_all[f] = nudata['RWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_CIWP_max[f] = nudata['CIWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_CIWP_mean[f] = nudata['CIWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_CIWP_mean_all[f] = nudata['CIWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_SIWP_max[f] = nudata['SIWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_SIWP_mean[f] = nudata['SIWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_SIWP_mean_all[f] = nudata['SIWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_GIWP_max[f] = nudata['GIWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_GIWP_mean[f] = nudata['GIWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_GIWP_mean_all[f] = nudata['GIWP'][0,:,:].where(this_feature, other = 0.0).mean().values
        
            feature_HIWP_max[f] = nudata['HIWP'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_HIWP_mean[f] = nudata['HIWP'][0,:,:].where(this_feature, other = float('nan')).mean().values
            feature_HIWP_mean_all[f] = nudata['HIWP'][0,:,:].where(this_feature, other = 0.0).mean().values
# 
            feature_FED[f] = nudata['LIGHTDENS'][0,:,:].where(this_feature, other = 0.0).max().values
            feature_SRC[f] = nudata['LIGHTDIS'][0,:,:].where(this_feature, other = 0.0).max().values

    
    
        print("Cleaning memory")
        del nudata
        gc.collect()

        # else:
            # continue
        
#     final_snapshot = tracemalloc.take_snapshot()

#     top_stats = final_snapshot.compare_to(prev_snapshot, 'lineno')

#     print("---------------------------------------------------------")
#     [print(stat) for stat in top_stats[:15]]
    
#     display_top(prev_snapshot)
#     display_top(final_snapshot)
    
    end = time.time()
    print(end - start)
    track_dim = "track"
    cell_dim = "cell"
    feature_dim = "feature"

    test = xr.Dataset(
        {
            "track": (track_dim, xrdata["track"].values),
            "cell": (cell_dim, xrdata["cell"].values),
            "feature": (feature_dim, xrdata["feature"].values),
            "feature_area": (feature_dim, (xrdata["feature_area"].values) * dxy * dxy),
            "feature_maxrefl": (feature_dim, xrdata["max_reflectivity"].values),
            "feature_TPW_max": (feature_dim, xr.DataArray(list(feature_TPW_max.values()), coords={'feature':list(feature_TPW_max.keys())}).data),
            "feature_LWP_max": (feature_dim, xr.DataArray(list(feature_LWP_max.values()), coords={'feature':list(feature_LWP_max.keys())}).data),
            "feature_IWP_max": (feature_dim, xr.DataArray(list(feature_IWP_max.values()), coords={'feature':list(feature_IWP_max.keys())}).data),
            "feature_CLWP_max": (feature_dim, xr.DataArray(list(feature_CLWP_max.values()), coords={'feature':list(feature_CLWP_max.keys())}).data),
            "feature_RWP_max": (feature_dim, xr.DataArray(list(feature_RWP_max.values()), coords={'feature':list(feature_RWP_max.keys())}).data),
            "feature_CIWP_max": (feature_dim, xr.DataArray(list(feature_CIWP_max.values()), coords={'feature':list(feature_CIWP_max.keys())}).data),
            "feature_SIWP_max": (feature_dim, xr.DataArray(list(feature_SIWP_max.values()), coords={'feature':list(feature_SIWP_max.keys())}).data),
            "feature_GIWP_max": (feature_dim, xr.DataArray(list(feature_GIWP_max.values()), coords={'feature':list(feature_GIWP_max.keys())}).data),
            "feature_HIWP_max": (feature_dim, xr.DataArray(list(feature_HIWP_max.values()), coords={'feature':list(feature_HIWP_max.keys())}).data),
            "feature_TPW_mean": (feature_dim, xr.DataArray(list(feature_TPW_mean.values()), coords={'feature':list(feature_TPW_mean.keys())}).data),
            "feature_LWP_mean": (feature_dim, xr.DataArray(list(feature_LWP_mean.values()), coords={'feature':list(feature_LWP_mean.keys())}).data),
            "feature_IWP_mean": (feature_dim, xr.DataArray(list(feature_IWP_mean.values()), coords={'feature':list(feature_IWP_mean.keys())}).data),
            "feature_CLWP_mean": (feature_dim, xr.DataArray(list(feature_CLWP_mean.values()), coords={'feature':list(feature_CLWP_mean.keys())}).data),
            "feature_RWP_mean": (feature_dim, xr.DataArray(list(feature_RWP_mean.values()), coords={'feature':list(feature_RWP_mean.keys())}).data),
            "feature_CIWP_mean": (feature_dim, xr.DataArray(list(feature_CIWP_mean.values()), coords={'feature':list(feature_CIWP_mean.keys())}).data),
            "feature_SIWP_mean": (feature_dim, xr.DataArray(list(feature_SIWP_mean.values()), coords={'feature':list(feature_SIWP_mean.keys())}).data),
            "feature_GIWP_mean": (feature_dim, xr.DataArray(list(feature_GIWP_mean.values()), coords={'feature':list(feature_GIWP_mean.keys())}).data),
            "feature_HIWP_mean": (feature_dim, xr.DataArray(list(feature_HIWP_mean.values()), coords={'feature':list(feature_HIWP_mean.keys())}).data),
            "feature_TPW_mean_all": (feature_dim, xr.DataArray(list(feature_TPW_mean_all.values()), coords={'feature':list(feature_TPW_mean_all.keys())}).data),
            "feature_LWP_mean_all": (feature_dim, xr.DataArray(list(feature_LWP_mean_all.values()), coords={'feature':list(feature_LWP_mean_all.keys())}).data),
            "feature_IWP_mean_all": (feature_dim, xr.DataArray(list(feature_IWP_mean_all.values()), coords={'feature':list(feature_IWP_mean_all.keys())}).data),
            "feature_CLWP_mean_all": (feature_dim, xr.DataArray(list(feature_CLWP_mean_all.values()), coords={'feature':list(feature_CLWP_mean_all.keys())}).data),
            "feature_RWP_mean_all": (feature_dim, xr.DataArray(list(feature_RWP_mean_all.values()), coords={'feature':list(feature_RWP_mean_all.keys())}).data),
            "feature_CIWP_mean": (feature_dim, xr.DataArray(list(feature_CIWP_mean_all.values()), coords={'feature':list(feature_CIWP_mean_all.keys())}).data),
            "feature_SIWP_mean_all_all": (feature_dim, xr.DataArray(list(feature_SIWP_mean_all.values()), coords={'feature':list(feature_SIWP_mean_all.keys())}).data),
            "feature_GIWP_mean_all": (feature_dim, xr.DataArray(list(feature_GIWP_mean_all.values()), coords={'feature':list(feature_GIWP_mean_all.keys())}).data),
            "feature_HIWP_mean_all": (feature_dim, xr.DataArray(list(feature_HIWP_mean_all.values()), coords={'feature':list(feature_HIWP_mean_all.keys())}).data),

            "feature_SRC": (feature_dim, xr.DataArray(list(feature_SRC.values()), coords={'feature':list(feature_SRC.keys())}).data),
            "feature_FED": (feature_dim, xr.DataArray(list(feature_FED.values()), coords={'feature':list(feature_FED.keys())}).data),
        }

    ).set_coords(['feature', 'cell', 'track'])
    
    
    # Clean up a couple other things. This could be fixed above, too...
    # test['feature_area'] = test.feature_area/(1.0*1.0) # Undo the conversion to km^2


    # -1 was being used insteaed of 0. For these variables, we want zero since it
    # indicates the value of the observed quantity, not the absence of a measurement.
    data_vars = [var for var in test.variables if 'feature_' in var]
    for var in data_vars:
        missing = test[var]<0
        test[var][missing] = 0
        
    # outfilename = "timeseries_data_melt{0}.nc".format(int(meltlev))
    outfilename = "timeseries_data_nuwrf.nc"#.format(int(meltlev))
  
    test.to_netcdf(os.path.join(savedir, outfilename))
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    import dask
    import xarray as xr
    xr.set_options(file_cache_maxsize=1)
    with dask.config.set(scheduler='single-threaded'):
        main(args)