#!/usr/bin/env python
# coding: utf-8

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
python knb_tobac_timseries.py --path="/Users/kelcy/DATA/20220322/" --lmapath="/Users/kelcy/DATA/PERiLS LMA deployments/V1.1_data/gridded/20220322_KGWX/" --tobacpath="/Users/kelcy/PYTHON/tracer-jcss_EBfork/tobac_Save_20220322/" --meltinglevel=4.0 --type="NEXRAD"


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
        "--meltinglevel",
        metavar="meltinglev",
        required=True,
        dest="meltinglev",
        action="store",
        help="Melting level in KM",
        type=float
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
    # Read in feature information
    savedir = args.tobacpath
    xrdata = xr.open_dataset(savedir + "Track_features_merges.nc")

    path = args.path + "*.nc"

    data = xr.open_mfdataset(path)
    data["time"].encoding["units"] = "seconds since 2000-01-01 00:00:00"
    nc_grid = load_cfradial_grids(path)
    nclon = nc_grid["point_longitude"][0, :, :].data
    nclat = nc_grid["point_latitude"][0, :, :].data

    ref = 10
    rhv = 0.9
    kdp_min = 0.75
    zdr_min = 1.0
    # This is the qc'd version of KDP and ZDR above threshold *regardless* of frame. It is computed lazily, so no penalty for doing it over the whole dataset.
    radar_good = (data["cross_correlation_ratio"] >= rhv) & (
        data["reflectivity"] >= ref
    )
    kdp_large = radar_good & (data["KDP_CSU"] >= kdp_min)
    kdp_qc = data["KDP_CSU"].where(kdp_large)
    zdr_large = radar_good & (data["differential_reflectivity"] >= zdr_min)
    zdr_qc = data["differential_reflectivity"].where(zdr_large)

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

    feature_area = np.zeros(len(xrdata["feature"].values))
    feature_area[:] = -1
    feature_area[:] = (xrdata["feature_area"].values) * dxy * dxy
    feature_maxrefl = np.zeros(len(xrdata["feature"].values))
    feature_maxrefl[:] = -1
    feature_maxrefl = xrdata["max_reflectivity"].values
    feature_zdrvol = np.zeros(len(xrdata["feature"].values))
    feature_zdrvol[:] = -1
    feature_kdpvol = np.zeros(len(xrdata["feature"].values))
    feature_kdpvol[:] = -1
    flash_count_arr = np.zeros(len(xrdata["feature"].values))
    flash_count_arr[:] = -1
    area_LE_4km = np.zeros(len(xrdata["feature"].values))
    area_LE_4km[:] = -1
    area_GT_4km = np.zeros(len(xrdata["feature"].values))
    area_GT_4km[:] = -1
    feature_zdrcol_max = np.zeros(len(xrdata["feature"].values))
    feature_kdpcol_max = np.zeros(len(xrdata["feature"].values))
    feature_zdrcol_max[:] = -1
    feature_kdpcol_max[:] = -1
    feature_zdrcol_mean = np.zeros(len(xrdata["feature"].values))
    feature_kdpcol_mean = np.zeros(len(xrdata["feature"].values))
    feature_zdrcol_mean[:] = -1
    feature_kdpcol_mean[:] = -1
    feature_zdrcol_total = np.zeros(len(xrdata["feature"].values))
    feature_kdpcol_total = np.zeros(len(xrdata["feature"].values))
    feature_zdrcol_total[:] = -1
    feature_kdpcol_total[:] = -1

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

    # Find the indicie in rhgt that correponds to the melting level given in the command line
    meltlev = args.meltinglev*1000.0
    ind2=np.argmin(np.absolute(rhgt - meltlev)) #8

    del data
    start = time.time()

    feature_area = (xrdata["feature_area"].values) * dxy * dxy
    feaure_maxrefl = xrdata["max_reflectivity"].values
    print(np.nanmax(np.unique(xrdata["feature_time_index"].values)))
    print(np.nanmax(xrdata["feature"].values))
    
    # tracemalloc.start()
    for num, i in enumerate(np.unique(xrdata["feature_time_index"].values)):
        # if i>10:
        #     break
        print(i)
#         if i < 350:
#             continue
        # if i==2:
        #     prev_snapshot = tracemalloc.take_snapshot()

        zdrvol = (zdr_qc[i, ind2 : ind2 + 6, :, :]).count(dim="z").compute()
        kdpvol = (kdp_qc[i, ind2 : ind2 + 6, :, :]).count(dim="z").compute()
        zdrcol = (zdr_qc[i, ind2 : ind2 + 6, :, :]).sum(dim="z").compute()
        kdpcol = (kdp_qc[i, ind2 : ind2 + 6, :, :]).sum(dim="z").compute()
        ids = np.where(xrdata["feature_time_index"].values == i)
        ids = xrdata["feature"].values[ids]
        features_this_frame = xrdata["segmentation_mask"].data[i, :, :]

        for nid, f in enumerate(ids):
            print(f)
            this_feature = (features_this_frame == f)
            feature_zdrvol[f - 1] = (
                zdrvol.where(this_feature, other=0).sum().values
                * grid_box_vol
            )
            feature_kdpvol[f - 1] = (
                kdpvol.where(this_feature, other=0).sum().values
                * grid_box_vol
            )
            feature_kdpcol_max[f - 1] = kdpcol.where(
                this_feature, other=0
            ).values.max()
            feature_zdrcol_max[f - 1] = zdrcol.where(
                this_feature, other=0
            ).values.max()
            feature_kdpcol_mean[f - 1] = np.nanmean(
                kdpcol.where(this_feature, other=np.NAN).values
            )
            feature_zdrcol_mean[f - 1] = np.nanmean(
                zdrcol.where(this_feature, other=np.NAN).values
            )
            feature_kdpcol_total[f - 1] = kdpcol.where(
                this_feature, other=0.0
            ).values.sum()
            feature_zdrcol_total[f - 1] = zdrcol.where(
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
                    flash_count_arr[f - 1] = flash_count / dt
                    area_LE_4km[f - 1] = flash_le4 / dt
                    area_GT_4km[f - 1] = flash_gt4 / dt
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
    track_dim = "tracks"
    cell_dim = "cells"
    feature_dim = "features"

    # Create the dataset and make the coordinate and dimension names match the ID variables
    test = xr.Dataset(
        {
            "track": (track_dim, xrdata["track"].values),
            "cell": (cell_dim, xrdata["cell"].values),
            "feature": (feature_dim, xrdata["feature"].values),
            "feature_area": (feature_dim, feature_area),
            "feature_maxrefl": (feature_dim, feature_maxrefl),
            "feature_zdrvol": (feature_dim, feature_zdrvol),
            "feature_kdpvol": (feature_dim, feature_kdpvol),
            "feature_zdrcol": (feature_dim, feature_zdrcol_max),
            "feature_kdpcol": (feature_dim, feature_kdpcol_max),
            "feature_zdrcol_mean": (feature_dim, feature_zdrcol_mean),
            "feature_kdpcol_mean": (feature_dim, feature_kdpcol_mean),
            "feature_zdrcol_total": (feature_dim, feature_zdrcol_total),
            "feature_kdpcol_total": (feature_dim, feature_kdpcol_total),
            "feature_flash_count": (feature_dim, flash_count_arr),
            "feature_area_LE4km": (feature_dim, area_LE_4km),
            "feature_area_GT_4km": (feature_dim, area_GT_4km),
        }
    ).swap_dims({'features':'feature', 'tracks':'track', 'cells':'cell'}
    ).set_coords(['feature', 'cell', 'track'])
    
    # Clean up a couple other things. This could be fixed above, too...
    test['feature_area'] = test.feature_area/(.5*.5) # Undo the conversion to km^2
    test['feature_flash_count_area_LE_4km'] = test.feature_area_LE4km
    test['feature_flash_count_area_GT_4km'] = test.feature_area_GT_4km
    test = test.drop_vars(('feature_area_LE4km','feature_area_GT_4km'))

    # -1 was being used insteaed of 0. For these variables, we want zero since it
    # indicates the value of the observed quantity, not the absence of a measurement.
    data_vars = [var for var in test.variables if 'feature_' in var]
    for var in data_vars:
        missing = test[var]<0
        test[var][missing] = 0
  
    test.to_netcdf(os.path.join(savedir, "timeseries_data.nc"))
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    import dask
    import xarray as xr
    xr.set_options(file_cache_maxsize=1)
    with dask.config.set(scheduler='single-threaded'):
        main(args)