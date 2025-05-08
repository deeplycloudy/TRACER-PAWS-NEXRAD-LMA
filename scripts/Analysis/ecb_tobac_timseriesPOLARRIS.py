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
python knb_tobac_timseriesPOLARRIS.py --polarrispath="/Users/kelcy/DATA/20220322/" --nuwrfpath="/Users/kelcy/DATA/PERiLS LMA deployments/V1.1_data/gridded/20220322_KGWX/" --tobacpath="/Users/kelcy/PYTHON/tracer-jcss_EBfork/tobac_Save_20220322/" --type="POLARRIS"


Variables
=========
About the variables created and/or used by this script:
feature_grid_cell_count - from tobac, maybe not reliable.
feature_area - count of grid boxes in the 2D feature footprint.
feature_maxrefl -  max reflectivity anywhere in 3D in feature.
feature_zdrvol - count of grid boxes Zdr above 1 dB, in 3 km slab above melting level in feature. Thresholds from van Lier Walqui.
feature_kdpvol - count of grid boxes Kdp above 0.75 dB, in 3 km slab above melting level in feature. Thresholds from van Lier Walqui.
feature_zdrcol - sum of values vertical dimension (in 3 km slab) above Zdr threshold, and then max anywhere in 2D feature. "Column strength."
feature_kdpcol - sum of values vertical dimension (in 3 km slab) above Kdp threshold, and then max anywhere in 2D feature. "Column strength."
feature_zdrcol_mean - sum of values vertical dimension (in 3 km slab) above Zdr threshold, and then average across 2D feature.
feature_kdpcol_mean - sum of values vertical dimension (in 3 km slab) above Kdp threshold, and then average across 2D feature.
feature_zdrcol_total - sum of values in vertical and horizontal dimensions above Zdr threshold over the whole 3D column feature.
feature_kdpcol_total - sum of values in vertical and horizontal dimensions above Kdp threshold over the whole 3D column feature.
feature_kdpwt_total - the total Kdp values in the feature, occurring between the melting level and 1km below the freezing level, weighted by its height above the melting level.
feature_zdrwt_total - the total Zdr values in the feature, occurring between the melting level and 1km below the freezing level, weighted by its height above the melting level.

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
        "--polarrispath",
        metavar="polarrispath",
        required=True,
        dest="polarrispath",
        action="store",
        help="path in which the data is located",
    )
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

import heapq
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
    #load reference lat lon grid for polarris data
    ncgrid = pyart.io.read_grid("/home/jovyan/efs/tracer/POLARRIS_GRIDDED/2022060400/VCP12_CfRadial_2022_0605_020221_grid.nc").to_xarray()
    nclon = ncgrid['lat'].data
    nclat = ncgrid['lon'].data 

    files = sorted(glob(args.nuwrfpath + "wrf2dout_d02_*"))
    time_arr = []
    for i in files:
        time_arr.append(pd.to_datetime(i[-19:],format = "%Y-%m-%d_%H:%M:%S"))
    time_arr = np.array(time_arr)
    nuwrf_df = pd.DataFrame(time_arr, columns=['nuwrf_time'])
    data_crs = ccrs.LambertConformal(central_longitude=-96.30000305175781,central_latitude = 29.500003814697266)

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
    track_dBZ_thresh = 35.0
    feature_is_bright = (xrdata['max_reflectivity'] > track_dBZ_thresh)
    reduced_track_ids = np.unique(xrdata[{'feature':feature_is_bright}].feature_parent_track_id)
    traversal = OneToManyTraversal(xrdata, ('track','cell','feature'), ('cell_parent_track_id', 'feature_parent_cell_id'))
    xrdata = traversal.reduce_to_entities('track', reduced_track_ids)
    print('reduced')
    
    #load in polarris data
    polpath = args.polarrispath + "*_grid.nc"

    pdata = xr.open_mfdataset(polpath,engine="netcdf4", combine="nested",  concat_dim="time")
    pdata["time"].encoding["units"] = "seconds since 2000-01-01 00:00:00"
    pfiles = sorted(glob(polpath))
    arr = []
    for i in pfiles:
        arr.append(pd.to_datetime(i[-24:-8],format = "%Y_%m%d_%H%M%S"))
    arr = pd.DatetimeIndex(arr)
    pdata = pdata.assign_coords(time=arr)
    # nc_grid = load_cfradial_grids(path)
    # nclon = nc_grid["point_longitude"][0, :, :].data
    # nclat = nc_grid["point_latitude"][0, :, :].data

    ref = 10
    rhv = 0.9
    rhv_col_thresh = 0.98
    kdp_min = 0.75
    zdr_min = 1.0
    # This is the qc'd version of KDP and ZDR above threshold *regardless* of frame. It is computed lazily, so no penalty for doing it over the whole dataset.
    #20230502: we have decided to not QC the ZDR/KDP in order to not remove areas where a KDP column is present with low Rho_hv values. 
    #radar_good = (data["cross_correlation_ratio"] >= rhv) & (
    #    data["reflectivity"] >= ref
    #)
    # 20230630 Add back in threshold on large reflectivity values to ensure no spurious values at cloud edge in low SNR.
    # Even if the 2D features are thresholdled on 15 dBZ, the actual 3D grids within those features can have
    # low reflectivity at cloud top.
    zh_large = (pdata["CZ"] >= ref)
    
    kdp_large = (pdata["KD"] >= kdp_min)
    kdp_qc = pdata["KD"].where(kdp_large & zh_large)
    zdr_large = (pdata["DR"] >= zdr_min)
    zdr_qc = pdata["DR"].where(zdr_large & zh_large)
    # June 30 2023: add RHV deficit. Deficit is the value below 1.0 of any values below the chosen threshold.
    # Guarantees positive values since we ensure we only use pixels with values less than 1.0.
    rhv_small = (pdata["RH"] <= rhv_col_thresh)
    rhv_deficit_qc = (1.0 - pdata["RH"]).where(zh_large & rhv_small)

    deltax = pdata["x"][1:].values - pdata["x"][:-1].values
    deltaz = pdata["z"][1:].values - pdata["z"][:-1].values
    dxy = np.abs(np.mean(deltax)) / 1000
    dz = np.abs(np.mean(deltaz)) / 1000
    print("Mean spacing of data in x:", dxy)
    print("Mean spacing of data in z:", dz)
    grid_box_vol = dz * dxy * dxy

    #val = pd.to_datetime(xrdata["time"].values)

    feature_zdrvol = dict()
    feature_kdpvol = dict()
    feature_rhvdeficitvol = dict()
    # flash_count_arr = dict()
    feature_SRC = dict()
    feature_FED = dict()
    feature_zdrcol_max = dict()
    feature_kdpcol_max = dict()
    feature_rhvdeficitcol_max = dict()
    feature_zdrcol_mean = dict()
    feature_kdpcol_mean = dict()
    feature_rhvdeficitcol_mean = dict()
    feature_zdrcol_total = dict()
    feature_kdpcol_total = dict()
    feature_rhvdeficitcol_total = dict()
    feature_zdrwcol_total = dict()
    feature_kdpwcol_total = dict()
    feature_rhvdeficitwcol_total = dict()
    x_mesh,z_mesh,y_mesh = np.meshgrid(pdata['x'],pdata['z'],pdata['y'])
    rhgt = (
        np.array(
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
        * 1000.0
    )


    del pdata
    start = time.time()
    if len(xrdata['feature']) == 0:
        print('no features')
    else:    
        print(np.nanmax(np.unique(xrdata["feature_time_index"].values)))
        print(np.nanmax(xrdata["feature"].values))
    
    # tracemalloc.start()
        for num, i in enumerate(np.unique(xrdata["feature_time_index"].values)):

            print(i)

            ids = np.where(xrdata["feature_time_index"].values == i)
            ids = xrdata["feature"].values[ids]
            features_this_frame = xrdata["segmentation_mask"].data[i, :, :]
            dilated_features_this_frame = dilation(xrdata["segmentation_mask"].data[i, :, :], disk(9))

    
            # t1 = pd.to_datetime(xrdata['feature_time_str'][ids].values[0],format = "%Y-%m-%d %H:%M:%S")
            t1 = pd.to_datetime(xrdata['time'][i].values)
            #print(t1)
            nuwrf_file = np.argmin(abs(nuwrf_df - t1))

            nudata = xr.open_mfdataset(files[nuwrf_file], engine="netcdf4",parallel=True,
                concat_dim="Time", combine="nested", chunks={"Time": 1},decode_times=False,
                drop_variables=drop_list).xwrf.postprocess()
            nudata = nudata.rename_dims({'Time': 'time'})
            nudata['time'] = nudata['Time']
            print(nudata['time'].values)
         # Find the index in rhgt that correponds to the melting level given in the command line
            meltlev = np.nanmean(nudata['FRZLVL'].values)#in meters already
            ind2=np.argmin(np.absolute(rhgt - meltlev)) #8
            frzlev = 10000.
            indfrz = np.argmin(np.absolute(rhgt - frzlev))
            # print(ind2)
            # print(indfrz)
 

        #     prev_snapshot = tracemalloc.take_snapshot()

            zdr_qc_here = zdr_qc[i, ind2 : indfrz, :, :].compute()
            kdp_qc_here = kdp_qc[i, ind2 : indfrz, :, :].compute()
            rhv_deficit_qc_here = rhv_deficit_qc[i, ind2 : indfrz, :, :].compute()
            z_weight_here = (z_mesh[ind2:indfrz,:,:]-rhgt[ind2])
            zdrvol = (zdr_qc_here).count(dim="z").compute()
            kdpvol = (kdp_qc_here).count(dim="z").compute()
            rhvdeficitvol = (rhv_deficit_qc_here).count(dim="z").compute()
            zdrcol = (zdr_qc_here).sum(dim="z").compute()
            kdpcol = (kdp_qc_here).sum(dim="z").compute()
            rhvdeficitcol = (rhv_deficit_qc_here).sum(dim="z").compute()
        # weights use the nearest grid box height to the melting level instead of the exact melting level to avoid negative height values.
            kdpwcol = (kdp_qc_here*z_weight_here).sum(dim="z").compute() 
            zdrwcol = (zdr_qc_here*z_weight_here).sum(dim="z").compute()
            rhvdeficitwcol = (rhv_deficit_qc_here*z_weight_here).sum(dim="z").compute()
        
       
            for nid, f in enumerate(ids):
                print(f)
                this_feature = (features_this_frame == f)
                feature_zdrvol[f] = (
                    zdrvol.where(this_feature, other=0.0).sum().values
                    * grid_box_vol)
                feature_kdpvol[f] = (
                    kdpvol.where(this_feature, other=0.0).sum().values
                    * grid_box_vol)
                feature_rhvdeficitvol[f] = (
                    rhvdeficitvol.where(this_feature, other=0.0).sum().values
                    * grid_box_vol)
                feature_kdpcol_max[f] = kdpcol.where(
                    this_feature, other=0.0).values.max()
                feature_zdrcol_max[f] = zdrcol.where(
                    this_feature, other=0.0).values.max()
                feature_rhvdeficitcol_max[f] = rhvdeficitcol.where(
                    this_feature, other=0.0).values.max()
                feature_kdpcol_mean[f] = np.nanmean(
                    kdpcol.where(this_feature, other=0.0).values)
                feature_zdrcol_mean[f] = np.nanmean(
                    zdrcol.where(this_feature, other=0.0).values)
                feature_rhvdeficitcol_mean[f] = np.nanmean(
                    rhvdeficitcol.where(this_feature, other=0.0).values)
                feature_kdpcol_total[f] = kdpcol.where(
                    this_feature, other=0.0).values.sum()
                feature_zdrcol_total[f] = zdrcol.where(
                    this_feature, other=0.0).values.sum()
                feature_rhvdeficitcol_total[f] = rhvdeficitcol.where(
                    this_feature, other=0.0).values.sum()
                feature_zdrwcol_total[f] = zdrwcol.where(
                    this_feature, other=0.0).values.sum()
                feature_kdpwcol_total[f] = kdpwcol.where(
                    this_feature, other=0.0).values.sum()
                feature_rhvdeficitwcol_total[f] = rhvdeficitwcol.where(
                    this_feature, other=0.0).values.sum()
                
                
            if np.nanmax(nudata['LIGHTDIS'].values) == 0:
                print('no ltg in timestep')
                for nid, f in enumerate(ids):
                    feature_FED[f] = 0
                    feature_SRC[f] = 0
                continue
            else:    
                for nid, f in enumerate(ids):
                    print(f)
                    #print(np.nanmax(nudata['LIGHTDIS'].values))
                    if np.nanmax(nudata['LIGHTDIS'].values) == 0:
                        print('no ltg in feature')
                        feature_FED[f] = 0
                        feature_SRC[f] = 0
                        continue

                    xy = np.where(dilated_features_this_frame == f)

            #Check if there are any features in the scene, if the feature exists
                    if len(xy[0]) <= 0:
                        feature_FED[f] = 0
                        feature_SRC[f] = 0
                        print('feature does not exist')
                        continue

            #Check if there is any lightning in the NUWRF data at the nearest time step

    
                    FED = []
                    SRC = []
                    dist = [] 
                    index = []
                #These are the ncgrid lat lon of the feature f into an array 
                    arrlat = nclat[xy[1]]
                    arrlon = nclon[xy[0]]

                    combined_x_y_arrays = np.dstack(
                        [np.array(arrlon).ravel(), np.array(arrlat).ravel()]
                        )[0]
                    kdtree = spatial.KDTree(combined_x_y_arrays)
        
            #These are the NUWRF points to test against the KD tree to find the nearest point:
                    out = data_crs.transform_points(ccrs.PlateCarree(),arrlat,arrlon)
    
                    da = nudata.sel(x = out[:,0], y = out[:,1],method = "nearest")
                    lon_points = da['XLONG'].values.ravel()
                    lat_points = da['XLAT'].values.ravel()
                    LGTDIS = da['LIGHTDIS'].values.ravel()
                    LGTDENS = da['LIGHTDENS'].values.ravel()
            #Test if there are any lightdis or dens vals in the scene
                    # if np.nanmax(da['LIGHTDIS']) ==0: 
                    #     feature_FED[f] = 0
                    #     feature_SRC[f] = 0
                    #     print('no ltg in this feature2')
            # continue
                    # else:
                        #First try the maximum point and find if it is in within 5km
                    N=5
                    mdisi = [list(LGTDIS).index(i) for i in heapq.nlargest(N, LGTDIS)]
                    mdensi = [list(LGTDENS).index(i) for i in heapq.nlargest(N, LGTDENS)]
                    for jj, lp in enumerate(mdisi):
                        print('doing mini comparison')
                        pt = [lat_points[lp], lon_points[lp]]
                        d, inder = kdtree.query(pt)
                        new_point = combined_x_y_arrays[inder]
                        index.append(inder)
                        d = great_circle(pt, new_point).km
                        dist.append(d)
                        if d <= 5:
                            feature_FED[f] = LGTDENS[lp]
                            feature_SRC[f] = LGTDIS[lp]
                            break
                    try:
                        feature_SRC[f]
                        
                    except:                
                        p =list(flatten(np.where(LGTDIS >0)))
                        pp = list(flatten(np.where(LGTDENS >0)))
                        p3 = list(set(p + pp))
                        LGTDIS = LGTDIS[p3]
                        LGTDENS = LGTDENS[p3]
                        lon_points = lon_points[p3]
                        lat_points = lat_points[p3]
                        print('starting point wise comparison')
                        for lp in range(len(lat_points)):
                            pt = [lat_points[lp], lon_points[lp]]
                            d, inder = kdtree.query(pt)
                            new_point = combined_x_y_arrays[inder]
                            index.append(inder)
                            d = great_circle(pt, new_point).km
                            dist.append(d)
            
                            if d > 5:
                                continue
                            else: 
                                FED.append(LGTDENS[lp])
                                SRC.append(LGTDIS[lp])
                        if (len(FED) > 0):
                            feature_FED[f] = np.nanmax(FED)
                        else:
                            feature_FED[f] = 0
                        if (len(SRC) > 0):
                            feature_SRC[f] = np.nanmax(SRC)
                        else:
                            feature_SRC[f] = 0

            print("Cleaning memory")
            del kdpvol, zdrvol, kdpcol, zdrcol, rhvdeficitcol,rhvdeficitwcol,rhv_deficit_qc_here
            try:
                nudata
                del nudata,da, dilated_features_this_frame,LGTDIS,LGTDENS
            except:
                print('no ltg var loaded') 
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
            "feature_zdrvol": (feature_dim, xr.DataArray(list(feature_zdrvol.values()), coords={'feature':list(feature_zdrvol.keys())}).data),
            "feature_kdpvol": (feature_dim, xr.DataArray(list(feature_kdpvol.values()), coords={'feature':list(feature_kdpvol.keys())}).data),
            "feature_rhvdeficitvol": (feature_dim, xr.DataArray(list(feature_rhvdeficitvol.values()), coords={'feature':list(feature_rhvdeficitvol.keys())}).data),
            "feature_zdrcol": (feature_dim, xr.DataArray(list(feature_zdrcol_max.values()), coords={'feature':list(feature_zdrcol_max.keys())}).data),
            "feature_kdpcol": (feature_dim, xr.DataArray(list(feature_kdpcol_max.values()), coords={'feature':list(feature_kdpcol_max.keys())}).data),
            "feature_rhvdeficitcol": (feature_dim, xr.DataArray(list(feature_rhvdeficitcol_max.values()), coords={'feature':list(feature_rhvdeficitcol_max.keys())}).data),
            "feature_zdrcol_mean": (feature_dim, xr.DataArray(list(feature_zdrcol_mean.values()), coords={'feature':list(feature_zdrcol_mean.keys())}).data),
            "feature_kdpcol_mean": (feature_dim, xr.DataArray(list(feature_kdpcol_mean.values()), coords={'feature':list(feature_kdpcol_mean.keys())}).data),
            "feature_rhvdeficitcol_mean": (feature_dim, xr.DataArray(list(feature_rhvdeficitcol_mean.values()), coords={'feature':list(feature_rhvdeficitcol_mean.keys())}).data),
            "feature_zdrcol_total": (feature_dim, xr.DataArray(list(feature_zdrcol_total.values()), coords={'feature':list(feature_zdrcol_total.keys())}).data),
            "feature_kdpcol_total": (feature_dim, xr.DataArray(list(feature_kdpcol_total.values()), coords={'feature':list(feature_kdpcol_total.keys())}).data),
            "feature_rhvdeficitcol_total": (feature_dim, xr.DataArray(list(feature_rhvdeficitcol_total.values()), coords={'feature':list(feature_rhvdeficitcol_total.keys())}).data),
            "feature_zdrwt_total": (feature_dim, xr.DataArray(list(feature_zdrwcol_total.values()), coords={'feature':list(feature_zdrwcol_total.keys())}).data),
            "feature_kdpwt_total": (feature_dim, xr.DataArray(list(feature_kdpwcol_total.values()), coords={'feature':list(feature_kdpwcol_total.keys())}).data),
            "feature_rhvdeficitwt_total": (feature_dim, xr.DataArray(list(feature_rhvdeficitwcol_total.values()), coords={'feature':list(feature_rhvdeficitwcol_total.keys())}).data),
            # "feature_flash_count": (feature_dim, xr.DataArray(list(flash_count_arr.values()), coords={'feature':list(flash_count_arr.keys())}).data),
            "feature_SRC": (feature_dim, xr.DataArray(list(feature_SRC.values()), coords={'feature':list(feature_SRC.keys())}).data),
            "feature_FED": (feature_dim, xr.DataArray(list(feature_FED.values()), coords={'feature':list(feature_FED.keys())}).data),
        }

    ).set_coords(['feature', 'cell', 'track'])
    
    
    # Clean up a couple other things. This could be fixed above, too...
    test['feature_area'] = test.feature_area/(.5*.5) # Undo the conversion to km^2
    # test['feature_flash_count_area_LE_4km'] = test.feature_area_LE_4km
    # test['feature_flash_count_area_GT_4km'] = test.feature_area_GT_4km
    # test = test.drop_vars(('feature_area_LE_4km','feature_area_GT_4km'))

    # -1 was being used insteaed of 0. For these variables, we want zero since it
    # indicates the value of the observed quantity, not the absence of a measurement.
    data_vars = [var for var in test.variables if 'feature_' in var]
    for var in data_vars:
        missing = test[var]<0
        test[var][missing] = 0
        
    # outfilename = "timeseries_data_melt{0}.nc".format(int(meltlev))
    outfilename = "timeseries_data_polarris.nc"#.format(int(meltlev))
  
    test.to_netcdf(os.path.join(savedir, outfilename))
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    import dask
    import xarray as xr
    xr.set_options(file_cache_maxsize=1)
    with dask.config.set(scheduler='single-threaded'):
        main(args)