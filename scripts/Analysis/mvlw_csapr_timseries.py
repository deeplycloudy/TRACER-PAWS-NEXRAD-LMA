#!/usr/bin/env python
# coding: utf-8

# MODIFIED 
# 20230502: KNB
# 20230630: ECB

import argparse


parse_desc = """For CSAPR RHIs, and the tracked cell (but not others), calculate column stats, including lightning, for a given day

The path is a string destination
Files in path must have a postfix of '.nc', this will be searched for internally.
Three paths are required: 
path: path to the radar data, currently only CSAPR data is supported in this version
lmapath: path to the lma flash sorted data
trackpath: path to the track csv file
type: Name of the type of data (NEXRAD/POLARRIS/NUWRF/CSAPR2) given as all uppercase string. Currently only CSAPR2 is supported.


Example
=======
python mvlw_csapr_timseries.py --path="/home/jovyan/efs/tracer/CSAPR2/oue_pi_csapr/houcsapr2pi/20220807/" --lmapath="/home/jovyan/efs/tracer/LIGHTNING/new_august/6sensor_minimum/" --trackpath="/home/jovyan/efs/tracer/cell-track-stats/data/houcsapr.20220807.csv" --meltinglevel=4.0 -- freezinglevel=9.9 --type="CSAPR2" 


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

# def display_top(snapshot, key_type='lineno', limit=10):
#     snapshot = snapshot.filter_traces((
#         tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#         tracemalloc.Filter(False, "<unknown>"),
#     ))
#     top_stats = snapshot.statistics(key_type)

#     print("Top %s lines" % limit)
#     for index, stat in enumerate(top_stats[:limit], 1):
#         frame = stat.traceback[0]
#         print("#%s: %s:%s: %.1f KiB"
#               % (index, frame.filename, frame.lineno, stat.size / 1024))
#         line = linecache.getline(frame.filename, frame.lineno).strip()
#         if line:
#             print('    %s' % line)

#     other = top_stats[limit:]
#     if other:
#         size = sum(stat.size for stat in other)
#         print("%s other: %.1f KiB" % (len(other), size / 1024))
#     total = sum(stat.size for stat in top_stats)
#     print("Total allocated size: %.1f KiB" % (total / 1024))



def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(
        "--path",
        metavar="path",
        required=True,
        dest="path",
        action="store",
        help="path in which the data is located",
    )
    parser.add_argument(
        "--lmapath",
        metavar="lmapath",
        required=True,
        dest="lmapath",
        action="store",
        help="path in which the flash sorted lma data is located",
    )
    parser.add_argument(
        "--trackpath",
        metavar="trackpath",
        required=True,
        dest="trackpath",
        action="store",
        help="path in which the track files are located",
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
        "--meltinglevel",
        metavar="meltinglev",
        required=True,
        dest="meltinglev",
        action="store",
        help="Melting level in KM",
        type=float
    )
    parser.add_argument(
        "--freezinglevel",
        metavar="freezinglev",
        required=True,
        dest="freezinglev",
        action="store",
        help="-40C level in KM",
        type=float
    )
    parser.add_argument(
        "--type",
        metavar="data type",
        required=False,
        dest="data_type",
        action="store",
        help="Datat name type, e.g., NEXRAD, POLARRIS, NUWRF, CSAPR2",
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

from pyxlma.lmalib.traversal import OneToManyTraversal

warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
warnings.filterwarnings("ignore", category=FutureWarning, append=True)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
import time


def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R=6370997.0):
    """
    Azimuthal equidistant Cartesian to geographic coordinate transform.

    Transform a set of Cartesian/Cartographic coordinates (x, y) to
    geographic coordinate system (lat, lon) using a azimuthal equidistant
    map projection [1]_.

    .. math::

        lat = \\arcsin(\\cos(c) * \\sin(lat_0) +
                       (y * \\sin(c) * \\cos(lat_0) / \\rho))

        lon = lon_0 + \\arctan2(
            x * \\sin(c),
            \\rho * \\cos(lat_0) * \\cos(c) - y * \\sin(lat_0) * \\sin(c))

        \\rho = \\sqrt(x^2 + y^2)

        c = \\rho / R

    Where x, y are the Cartesian position from the center of projection;
    lat, lon the corresponding latitude and longitude; lat_0, lon_0 are the
    latitude and longitude of the center of the projection; R is the radius of
    the earth (defaults to ~6371 km). lon is adjusted to be between -180 and
    180.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the same units as R, typically meters.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the projection.
    R : float, optional
        Earth radius in the same units as x and y. The default value is in
        units of meters.

    Returns
    -------
    lon, lat : array
        Longitude and latitude of Cartesian coordinates in degrees.

    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
        Survey Professional Paper 1395, 1987, pp. 191-202.

    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    rho = np.sqrt(x * x + y * y)
    c = rho / R

    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat_0_rad) + y * np.sin(c) * np.cos(lat_0_rad) / rho
        )
    lat_deg = np.rad2deg(lat_rad)
    # fix cases where the distance from the center of the projection is zero
    lat_deg[rho == 0] = lat_0

    x1 = x * np.sin(c)
    x2 = rho * np.cos(lat_0_rad) * np.cos(c) - y * np.sin(lat_0_rad) * np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)
    lon_deg = np.rad2deg(lon_rad)
    # Longitudes should be from -180 to 180 degrees
    lon_deg[lon_deg > 180] -= 360.0
    lon_deg[lon_deg < -180] += 360.0

    return lon_deg, lat_deg


def cartesian_to_geographic(grid_ds):
    """
    Cartesian to Geographic coordinate transform.

    Transform a set of Cartesian/Cartographic coordinates (x, y) to a
    geographic coordinate system (lat, lon) using pyproj or a build in
    Azimuthal equidistant projection.

    Parameters
    ----------
    grid_ds: xarray DataSet
        Cartesian coordinates in meters unless R is defined in different units
        in the projparams parameter.

    Returns
    -------
    lon, lat : array
        Longitude and latitude of the Cartesian coordinates in degrees.

    """
    projparams = grid_ds.ProjectionCoordinateSystem
    x = grid_ds.x.values
    y = grid_ds.y.values
    z = grid_ds.z.values
    z, y, x = np.meshgrid(z, y, x, indexing="ij")
    if projparams.attrs["grid_mapping_name"] == "azimuthal_equidistant":
        # Use Py-ART's Azimuthal equidistance projection
        lat_0 = projparams.attrs["latitude_of_projection_origin"]
        lon_0 = projparams.attrs["longitude_of_projection_origin"]
        if "semi_major_axis" in projparams:
            R = projparams.attrs["semi_major_axis"]
            lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R)
        else:
            lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0)
    else:
        # Use pyproj for the projection
        # check that pyproj is available
        if not _PYPROJ_AVAILABLE:
            raise MissingOptionalDependency(
                "PyProj is required to use cartesian_to_geographic "
                "with a projection other than pyart_aeqd but it is not "
                "installed"
            )
        proj = pyproj.Proj(projparams)
        lon, lat = proj(x, y, inverse=True)
    return lon, lat


def add_lat_lon_grid(grid_ds):
    lon, lat = cartesian_to_geographic(grid_ds)
    grid_ds["point_latitude"] = xr.DataArray(lat, dims=["z", "y", "x"])
    grid_ds["point_latitude"].attrs["long_name"] = "Latitude"
    grid_ds["point_latitude"].attrs["units"] = "degrees"
    grid_ds["point_longitude"] = xr.DataArray(lon, dims=["z", "y", "x"])
    grid_ds["point_longitude"].attrs["long_name"] = "Latitude"
    grid_ds["point_longitude"].attrs["units"] = "degrees"
    return grid_ds


def parse_grid_datetime(my_ds):
    year = my_ds["time"].dt.year
    month = my_ds["time"].dt.month
    day = my_ds["time"].dt.day
    hour = my_ds["time"].dt.hour
    minute = my_ds["time"].dt.minute
    second = my_ds["time"].dt.second
    return datetime(
        year=year, month=month, day=day, hour=hour, minute=minute, second=second
    )


def load_cfradial_grids(file_list):
    ds = xr.open_mfdataset(file_list)
    # Check for CF/Radial conventions
    if not ds.attrs["Conventions"] == "CF/Radial instrument_parameters":
        ds.close()
        raise IOError("TINT module is only compatible with CF/Radial files!")
    ds = add_lat_lon_grid(ds)

    return ds

def main(args):
    ###LOAD DATA
    lmafiles = sorted(glob(args.lmapath + "*.nc"))
    lma_longitude = []
    lma_latitude = []
    lma_flash_time = []
    lma_area = []
    for i, j in enumerate(lmafiles):
        print(j)
        lmadata1 = xr.open_dataset(j)
        lmadata1 = lmadata1.rename({"grid_time": "time"})
        if i == 0:
            lma_flash_time = list(pd.to_datetime(lmadata1["flash_time_start"].values))
        else:
            lma_flash_time = np.append(
                lma_flash_time,
                list(pd.to_datetime(lmadata1["flash_time_start"].values)),
            )
        lma_longitude = np.append(
            lma_longitude, lmadata1["flash_center_longitude"].values
        )
        lma_latitude = np.append(lma_latitude, lmadata1["flash_center_latitude"].values)
        lma_area = np.append(lma_area, lmadata1["flash_area"].values)
    lma_times = pd.to_datetime(lma_flash_time)

    time_sec = lma_times.hour * 3600 + lma_times.minute * 60 + lma_times.second
    
    #..Read in track information
    savedir = args.trackpath
    xrdata  = xr.open_dataset(savedir)
    
    #..Just look at literally everything
    feature_is_rhi = (xrdata['scan_mode'] == 'rhi')
    reduced_scans  = xrdata[feature_is_rhi]
    
    path = args.path + '*.nc'
    data = xr.open_mfdataset(path)
    data["time"].encoding["units"] =  "seconds since 2000-01-01 00:00:00"
    
    #..Skip processing for now, just use Oue specific_differential_phase_2. In the future, should add in 
    #..phase processed using CSU w/parameters = ( kdp_window=1.8, thresh_sdp=20.0, speckle=3 )
    
    #..Do grids (2d horiz-vert) for all?
    r_rez = .25
    z_rez = .25
    rgrd = 1000.*np.arange(0.,50.5,r_rez)
    zgrd = 1000.*np.arange(0.,15.5,z_rez)
    RGRD,ZGRD = np.meshgrid(rgrd,zgrd)
    melt_zidx = min(range(len(zgrd)), key=lambda i: abs(zgrd[i]-1000.*meltinglevel))
    frez_zidx = min(range(len(zgrd)), key=lambda i: abs(zgrd[i]-1000.*freezinglevel -1000.))  #..1km down from freez level
    
    #..Do loop through track times, grid, process and save to whatever
    for num, i in enumerate(np.unique(xrdata["time"].values)):

        print(i)
        
        #..Find the CSAPR file that matches this
        i
        
        #..Define time to start looking for lightning, time to stop looking for lightning
        
        #..Grid the CSAPR fields
        
        #..Cacalate column statistics (do we look for column boundaries?)
        
        #..Do lightning stats:
        time1 = time1_arr[i]
        time2 = time2_arr[i]
        dt = (time2 - time1) / 60

        iltg = np.squeeze(np.array(np.where(time_sec <= time2)))
        iiltg = np.squeeze(np.array(np.where(time_sec[iltg] > time1)))
        if iltg.size == 1:
            inds = iltg
        else:
            inds = iltg[iiltg]

        arrlat = []
        arrlon = []
        areas = lma_area[inds]
        if np.size(areas) == 1:
            areas = [areas]
        flash_le4 = 0
        flash_gt4 = 0
        flash_count = 0
        flash_count_list = []
        LE_4km = []
        GT_4km = []
        dist = []

        xy = np.where(features_this_frame == f)  # t,y,x

        if len(xy[0]) <= 0:
            flash_count_arr[f] = 0
            area_LE_4km[f] = 0
            area_GT_4km[f] = 0
            continue
        else:
            for ndd in range(len(xy[0])):
                arrlat.append(nclat[xy[0][ndd], xy[1][ndd]])
                arrlon.append(nclon[xy[0][ndd], xy[1][ndd]])
            combined_x_y_arrays = np.dstack(
                [np.array(arrlat).ravel(), np.array(arrlon).ravel()]
            )[0]

            kdtree = spatial.KDTree(combined_x_y_arrays)
            lon_points = np.atleast_1d(lma_longitude[inds])
            lat_points = np.atleast_1d(lma_latitude[inds])

            for lp in range(len(lat_points)):
                pt = [lat_points[lp], lon_points[lp]]
                d, inder = kdtree.query(pt)
                new_point = combined_x_y_arrays[inder]
                #             index.append(inder)
                d = great_circle(pt, new_point).km
                dist.append(d)
                if d > 3:
                    continue
                else:
                    if np.sqrt(areas[lp]) < 0.005:
                        continue
                    else:
                        flash_count += 1

                    if np.sqrt(areas[lp]) <= 4.0:  # might be inder
                        flash_le4 += 1
                    else:
                        flash_gt4 += 1

            if flash_count > 0:
                flash_count_arr[f] = flash_count / dt
                area_LE_4km[f] = flash_le4 / dt
                area_GT_4km[f] = flash_gt4 / dt
            else:
                flash_count_arr[f] = 0
                area_LE_4km[f] = 0
                area_GT_4km[f] = 0
    
    #-------------------------------------------------------------------------------------------------
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
    path = args.path + "*grid.nc"

    data = xr.open_mfdataset(path)
    data["time"].encoding["units"] = "seconds since 2000-01-01 00:00:00"
    nc_grid = load_cfradial_grids(path)
    nclon = nc_grid["point_longitude"][0, :, :].data
    nclat = nc_grid["point_latitude"][0, :, :].data

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
    zh_large = (data["reflectivity"] >= ref)
    
    kdp_large = (data["KDP_CSU"] >= kdp_min)
    kdp_qc = data["KDP_CSU"].where(kdp_large & zh_large)
    zdr_large = (data["differential_reflectivity"] >= zdr_min)
    zdr_qc = data["differential_reflectivity"].where(zdr_large & zh_large)
    # June 30 2023: add RHV deficit. Deficit is the value below 1.0 of any values below the chosen threshold.
    # Guarantees positive values since we ensure we only use pixels with values less than 1.0.
    rhv_small = (data["cross_correlation_ratio"] <= rhv_col_thresh)
    rhv_deficit_qc = (1.0 - data["cross_correlation_ratio"]).where(zh_large & rhv_small)

    deltax = data["x"][1:].values - data["x"][:-1].values
    deltaz = data["z"][1:].values - data["z"][:-1].values
    dxy = np.abs(np.mean(deltax)) / 1000
    dz = np.abs(np.mean(deltaz)) / 1000
    print("Mean spacing of data in x:", dxy)
    print("Mean spacing of data in z:", dz)
    grid_box_vol = dz * dxy * dxy

    val = pd.to_datetime(xrdata["time"].values)

    time1_arr = np.zeros(len(val))
    time2_arr = np.zeros(len(val))

    for i in range(len(val)):
        if i == 0:
            time1_arr[i] = val[0].hour * 3600 + val[0].minute * 60 + val[0].second - 300
            time2_arr[i] = val[0].hour * 3600 + val[0].minute * 60 + val[0].second
        else:
            time1_arr[i] = (
                val[i - 1].hour * 3600 + val[i - 1].minute * 60 + val[i - 1].second
            )
            time2_arr[i] = val[i].hour * 3600 + val[i].minute * 60 + val[i].second
            if time1_arr[i] > time2_arr[i]:
                time1_arr[i] = 0


    feature_zdrvol = dict()
    feature_kdpvol = dict()
    feature_rhvdeficitvol = dict()
    flash_count_arr = dict()
    area_LE_4km = dict()
    area_GT_4km = dict()
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
    x_mesh,z_mesh,y_mesh = np.meshgrid(data['x'],data['z'],data['y'])
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

    # Find the index in rhgt that correponds to the melting level given in the command line
    meltlev = args.meltinglev*1000.0
    ind2=np.argmin(np.absolute(rhgt - meltlev)) #8
    frzlev = args.freezinglev*1000.0 - 1000.0
    indfrz = np.argmin(np.absolute(rhgt - frzlev))
    del data
    start = time.time()
     
    print(np.nanmax(np.unique(xrdata["feature_time_index"].values)))
    print(np.nanmax(xrdata["feature"].values))
    
    # tracemalloc.start()
    for num, i in enumerate(np.unique(xrdata["feature_time_index"].values)):

        print(i)

        # if i==2:
        #     prev_snapshot = tracemalloc.take_snapshot()
        
        # By this point, we only have one time and the vertical dimension is only the mixed phase.
        # That should fit in memory with no problem.
        # Force compute to put these grids in memory for fast reuse in later steps.
        # This was not too slow before we added rhvdeficit, but the subtraction from 1 seems
        # to have really slowed it down.
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
        
        ids = np.where(xrdata["feature_time_index"].values == i)
        ids = xrdata["feature"].values[ids]
        features_this_frame = xrdata["segmentation_mask"].data[i, :, :]
        
        
        for nid, f in enumerate(ids):
            print(f)
            this_feature = (features_this_frame == f)

            
            feature_zdrvol[f] = (
                zdrvol.where(this_feature, other=0.0).sum().values
                * grid_box_vol
            )

            feature_kdpvol[f] = (
                kdpvol.where(this_feature, other=0.0).sum().values
                * grid_box_vol
            )

            feature_rhvdeficitvol[f] = (
                rhvdeficitvol.where(this_feature, other=0.0).sum().values
                * grid_box_vol
            )

            feature_kdpcol_max[f] = kdpcol.where(
                this_feature, other=0.0
            ).values.max()
            
            feature_zdrcol_max[f] = zdrcol.where(
                this_feature, other=0.0
            ).values.max()
            
            feature_rhvdeficitcol_max[f] = rhvdeficitcol.where(
                this_feature, other=0.0
            ).values.max()
            

            feature_kdpcol_mean[f] = np.nanmean(
                kdpcol.where(this_feature, other=0.0).values
            )
            
            feature_zdrcol_mean[f] = np.nanmean(
                zdrcol.where(this_feature, other=0.0).values
            )
            
            feature_rhvdeficitcol_mean[f] = np.nanmean(
                rhvdeficitcol.where(this_feature, other=0.0).values
            )

            feature_kdpcol_total[f] = kdpcol.where(
                this_feature, other=0.0
            ).values.sum()

            feature_zdrcol_total[f] = zdrcol.where(
                this_feature, other=0.0
            ).values.sum()

            feature_rhvdeficitcol_total[f] = rhvdeficitcol.where(
                this_feature, other=0.0
            ).values.sum()

            feature_zdrwcol_total[f] = zdrwcol.where(
                this_feature, other=0.0
            ).values.sum()
            
            feature_kdpwcol_total[f] = kdpwcol.where(
                this_feature, other=0.0
            ).values.sum()
            
            feature_rhvdeficitwcol_total[f] = rhvdeficitwcol.where(
                this_feature, other=0.0
            ).values.sum()

            time1 = time1_arr[i]
            time2 = time2_arr[i]
            dt = (time2 - time1) / 60

            iltg = np.squeeze(np.array(np.where(time_sec <= time2)))
            iiltg = np.squeeze(np.array(np.where(time_sec[iltg] > time1)))
            if iltg.size == 1:
                inds = iltg
            else:
                inds = iltg[iiltg]

            arrlat = []
            arrlon = []
            areas = lma_area[inds]
            if np.size(areas) == 1:
                areas = [areas]
            flash_le4 = 0
            flash_gt4 = 0
            flash_count = 0
            flash_count_list = []
            LE_4km = []
            GT_4km = []
            dist = []

            xy = np.where(features_this_frame == f)  # t,y,x

            if len(xy[0]) <= 0:
                flash_count_arr[f] = 0
                area_LE_4km[f] = 0
                area_GT_4km[f] = 0
                continue
            else:
                for ndd in range(len(xy[0])):
                    arrlat.append(nclat[xy[0][ndd], xy[1][ndd]])
                    arrlon.append(nclon[xy[0][ndd], xy[1][ndd]])
                combined_x_y_arrays = np.dstack(
                    [np.array(arrlat).ravel(), np.array(arrlon).ravel()]
                )[0]

                kdtree = spatial.KDTree(combined_x_y_arrays)
                lon_points = np.atleast_1d(lma_longitude[inds])
                lat_points = np.atleast_1d(lma_latitude[inds])

                for lp in range(len(lat_points)):
                    pt = [lat_points[lp], lon_points[lp]]
                    d, inder = kdtree.query(pt)
                    new_point = combined_x_y_arrays[inder]
                    #             index.append(inder)
                    d = great_circle(pt, new_point).km
                    dist.append(d)
                    if d > 3:
                        continue
                    else:
                        if np.sqrt(areas[lp]) < 0.005:
                            continue
                        else:
                            flash_count += 1

                        if np.sqrt(areas[lp]) <= 4.0:  # might be inder
                            flash_le4 += 1
                        else:
                            flash_gt4 += 1

                if flash_count > 0:
                    flash_count_arr[f] = flash_count / dt
                    area_LE_4km[f] = flash_le4 / dt
                    area_GT_4km[f] = flash_gt4 / dt
                else:
                    flash_count_arr[f] = 0
                    area_LE_4km[f] = 0
                    area_GT_4km[f] = 0
        print("Cleaning memory")
        del kdpvol, zdrvol, kdpcol, zdrcol
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
            "feature_flash_count": (feature_dim, xr.DataArray(list(flash_count_arr.values()), coords={'feature':list(flash_count_arr.keys())}).data),
            "feature_area_LE_4km": (feature_dim, xr.DataArray(list(area_LE_4km.values()), coords={'feature':list(area_LE_4km.keys())}).data),
            "feature_area_GT_4km": (feature_dim, xr.DataArray(list(area_GT_4km.values()), coords={'feature':list(area_GT_4km.keys())}).data),
        }

    ).set_coords(['feature', 'cell', 'track'])
    
    
    # Clean up a couple other things. This could be fixed above, too...
    test['feature_area'] = test.feature_area/(.5*.5) # Undo the conversion to km^2
    test['feature_flash_count_area_LE_4km'] = test.feature_area_LE_4km
    test['feature_flash_count_area_GT_4km'] = test.feature_area_GT_4km
    test = test.drop_vars(('feature_area_LE_4km','feature_area_GT_4km'))

    # -1 was being used insteaed of 0. For these variables, we want zero since it
    # indicates the value of the observed quantity, not the absence of a measurement.
    data_vars = [var for var in test.variables if 'feature_' in var]
    for var in data_vars:
        missing = test[var]<0
        test[var][missing] = 0
        
    outfilename = "timeseries_data_melt{0}.nc".format(int(meltlev))
  
    test.to_netcdf(os.path.join(savedir, outfilename))
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    import dask
    import xarray as xr
    xr.set_options(file_cache_maxsize=1)
    with dask.config.set(scheduler='single-threaded'):
        main(args)