#!/usr/bin/env python
# coding: utf-8

import argparse

parse_desc = """Find features, track, and plot all nexrad data in a given destination folder

The path is a string destination
Files in path must have a postfix of 'grid.nc'. 
threshold is the tracking threshold in dbz
speed is the tracking speed in tobac units. 
Site is a string NEXRAD location
Options for type are: NUWRF, POLARRIS, NEXRAD, and POLARRIS2
POLARRIS2 is reserved for 1min pre-gridded POLARRIS output for thermal code

Example
=======
python knb_tobac_plotting.py --path="/archive/TRACER_processing/JUNE/20220602/" 
        --tobacpath="/archive/TRACER_processing/JUNE/20220602/tobac_Save_20220602/" --type='NEXRAD' --lat=29.4719 --lon=-95.0792 --dxy=0.2


"""

def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument('--path',metavar='path', required=True,dest='path',
                        action = 'store',help='path in which the data is located')
    parser.add_argument('-o', '--output_path',
                        metavar='filename template including path',
                        required=False, dest='outdir', action='store',
                        default='.', help='path in which the data is located')
    parser.add_argument('--tobacpath', metavar='tobacpath', required=True,
                        dest='tobacpath', action='store',
                        help='Path to the tobac tracking files')

    parser.add_argument('--type', metavar='data type', required=True,
                        dest='data_type', action='store',
                        help='Datat name type, e.g., NEXRAD, POLARRIS, NUWRF')
    parser.add_argument('--lat', metavar='lat location for plot', required=True,
                        dest='plot_lat', action='store',type=float, 
                        help='The latitude center for plotting')
    parser.add_argument('--lon', metavar='lon location for plot', required=True,
                        dest='plot_lon', action='store',type=float, 
                        help='The longitude center for plotting')
    parser.add_argument('--dxy',metavar='xy grid spacing in km', required = False,
                        dest='dxy',action='store',type=float)
    return parser

# End parsing #


# Import libraries:
import xarray as xr
import numpy as np
import pandas as pd
import os
from six.moves import urllib
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import pyart
from datetime import datetime
import math
from pandas.core.common import flatten


try:
    import pyproj

    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False


# get_ipython().run_line_magic("matplotlib", "inline")
# %matplotlib widget
import tobac
from tobac.merge_split import merge_split_MEST
# from tobac.utils import standardize_track_dataset
# from tobac.utils import compress_all

# Disable a couple of warnings:
import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
warnings.filterwarnings("ignore", category=FutureWarning, append=True)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)



def standardize_track_dataset(TrackedFeatures, Mask, Projection=None):
    """
    Combine a feature mask with the feature data table into a common dataset.

    returned by tobac.segmentation
    with the TrackedFeatures dataset returned by tobac.linking_trackpy.

    Also rename the variables to be more desciptive and comply with cf-tree.

    Convert the default cell parent ID  to an integer table.

    Add a cell dimension to reflect

    Projection is an xarray DataArray

    TODO: Add metadata attributes

    Parameters
    ----------
    TrackedFeatures : xarray.core.dataset.Dataset
        xarray dataset of tobac Track information, the xarray dataset returned by tobac.tracking.linking_trackpy

    Mask: xarray.core.dataset.Dataset
        xarray dataset of tobac segmentation mask information, the xarray dataset returned
        by tobac.segmentation.segmentation


    Projection : xarray.core.dataarray.DataArray, default = None
        array.DataArray of the original input dataset (gridded nexrad data for example).
        If using gridded nexrad data, this can be input as: data['ProjectionCoordinateSystem']
        An example of the type of information in the dataarray includes the following attributes:
        latitude_of_projection_origin :29.471900939941406
        longitude_of_projection_origin :-95.0787353515625
        _CoordinateTransformType :Projection
        _CoordinateAxes :x y z time
        _CoordinateAxesTypes :GeoX GeoY Height Time
        grid_mapping_name :azimuthal_equidistant
        semi_major_axis :6370997.0
        inverse_flattening :298.25
        longitude_of_prime_meridian :0.0
        false_easting :0.0
        false_northing :0.0

    Returns
    -------

    ds : xarray.core.dataset.Dataset
        xarray dataset of merged Track and Segmentation Mask datasets with renamed variables.

    """
    import xarray as xr

    feature_standard_names = {
        # new variable name, and long description for the NetCDF attribute
        "frame": (
            "feature_time_index",
            "positional index of the feature along the time dimension of the mask, from 0 to N-1",
        ),
        "hdim_1": (
            "feature_hdim1_coordinate",
            "position of the feature along the first horizontal dimension in grid point space; a north-south coordinate for dim order (time, y, x)."
            "The numbering is consistent with positional indexing of the coordinate, but can be"
            "fractional, to account for a centroid not aligned to the grid.",
        ),
        "hdim_2": (
            "feature_hdim2_coordinate",
            "position of the feature along the second horizontal dimension in grid point space; an east-west coordinate for dim order (time, y, x)"
            "The numbering is consistent with positional indexing of the coordinate, but can be"
            "fractional, to account for a centroid not aligned to the grid.",
        ),
        "idx": ("feature_id_this_frame",),
        "num": (
            "feature_grid_cell_count",
            "Number of grid points that are within the threshold of this feature",
        ),
        "threshold_value": (
            "feature_threshold_max",
            "Feature number within that frame; starts at 1, increments by 1 to the number of features for each frame, and resets to 1 when the frame increments",
        ),
        "feature": (
            "feature",
            "Unique number of the feature; starts from 1 and increments by 1 to the number of features",
        ),
        "time": (
            "feature_time",
            "time of the feature, consistent with feature_time_index",
        ),
        "timestr": (
            "feature_time_str",
            "String representation of the feature time, YYYY-MM-DD HH:MM:SS",
        ),
        "projection_y_coordinate": (
            "feature_projection_y_coordinate",
            "y position of the feature in the projection given by ProjectionCoordinateSystem",
        ),
        "projection_x_coordinate": (
            "feature_projection_x_coordinate",
            "x position of the feature in the projection given by ProjectionCoordinateSystem",
        ),
        "lat": ("feature_latitude", "latitude of the feature"),
        "lon": ("feature_longitude", "longitude of the feature"),
        "ncells": (
            "feature_ncells",
            "number of grid cells for this feature (meaning uncertain)",
        ),
        "areas": ("feature_area",),
        "isolated": ("feature_isolation_flag",),
        "num_objects": ("number_of_feature_neighbors",),
        "cell": ("feature_parent_cell_id",),
        "time_cell": ("feature_parent_cell_elapsed_time",),
        "segmentation_mask": ("2d segmentation mask",),
    }
    new_feature_var_names = {
        k: feature_standard_names[k][0]
        for k in feature_standard_names.keys()
        if k in TrackedFeatures.variables.keys()
    }

    #     TrackedFeatures = TrackedFeatures.drop(["cell_parent_track_id"])
    # Combine Track and Mask variables. Use the 'feature' variable as the coordinate variable instead of
    # the 'index' variable and call the dimension 'feature'
    ds = xr.merge(
        [
            TrackedFeatures.swap_dims({"index": "feature"})
            .drop("index")
            .rename_vars(new_feature_var_names),
            Mask,
        ]
    )

    # Add the projection data back in
    if Projection is not None:
        ds["ProjectionCoordinateSystem"] = Projection

    return ds


def compress_all(nc_grids, min_dims=2, comp_level=4):
    """
    The purpose of this subroutine is to compress the netcdf variables as they are saved.
    This does not change the data, but sets netcdf encoding parameters.
    We allocate a minimum number of dimensions as variables with dimensions
    under the minimum value do not benefit from tangibly from this encoding.

    Parameters
    ----------
    nc_grids : xarray.core.dataset.Dataset
        Xarray dataset that is intended to be exported as netcdf

    min_dims : integer
        The minimum number of dimesnions, in integer value, a variable must have in order
        set the netcdf compression encoding.
    comp_level : integer
        The level of compression. Default values is 4.

    Returns
    -------
    nc_grids : xarray.core.dataset.Dataset
        Xarray dataset with netcdf compression encoding for variables with two (2) or more dimensions

    """

    for var in nc_grids:
        if len(nc_grids[var].dims) >= min_dims:
            # print("Compressing ", var)
            nc_grids[var].encoding["zlib"] = True
            nc_grids[var].encoding["complevel"] = comp_level
            nc_grids[var].encoding["contiguous"] = False
    return nc_grids


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


""" X-Array based TINT I/O module. """

import xarray as xr
import random
import numpy as np
import pyproj

# from .grid_utils import add_lat_lon_grid
from datetime import datetime


def load_cfradial_grids(file_list):
    ds = xr.open_mfdataset(file_list)
    # Check for CF/Radial conventions
    if not ds.attrs["Conventions"] == "CF/Radial instrument_parameters":
        ds.close()
        raise IOError("TINT module is only compatible with CF/Radial files!")
    ds = add_lat_lon_grid(ds)

    return ds
    
def load_cfradial_grids_polarris(ds_data):

    ds = add_lat_lon_grid(ds_data)

    return ds




# USING BOTH_DS XARRAY COMBINED DATA SET

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid


def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end


def plot(t_index, xrdata, max_refl, ncgrid, grid_lat, grid_lon, ind=None):
    # Get the data
    hsv_ctr_lat, hsv_ctr_lon = grid_lat, grid_lon #29.4719, -95.0792
   # hsv_ctr_lat, hsv_ctr_lon = 34.93055725, -86.08361053
    #     hsv_ctr_lat, hsv_ctr_lon = 33.89691544, -88.32919312

    refl = max_refl[t_index, :, :]

    t_step = str(ncgrid["time"][t_index].values)
    nclons = ncgrid["point_longitude"][0, :, :].data
    nclats = ncgrid["point_latitude"][0, :, :].data

    fname = "shapefiles/ne_10m_admin_1_states_provinces_lines.shp"
#     fname = "ne_10m_admin_1_states_provinces_lines.shp"
    # Plot
    # fig.clear()
    latlon_proj = ccrs.PlateCarree()
    cs_attrs = ncgrid["ProjectionCoordinateSystem"][0].attrs
    if cs_attrs["grid_mapping_name"] == "azimuthal_equidistant":
        grid_proj = ccrs.AzimuthalEquidistant(
            central_latitude=cs_attrs["latitude_of_projection_origin"],
            central_longitude=cs_attrs["longitude_of_projection_origin"],
            false_easting=cs_attrs["false_easting"],
            false_northing=cs_attrs["false_northing"],
        )
    projection = grid_proj
    axes_class = (GeoAxes, dict(map_projection=projection))
    axs = AxesGrid(
        fig,
        111,
        axes_class=axes_class,
        nrows_ncols=(1, 1),
        axes_pad=0.4,
        cbar_location="right",
        cbar_mode="each",
        cbar_pad=0.4,
        cbar_size="3%",
        label_mode="",
    )  # note the empty label_mode
    for ax in axs:
        ax.coastlines()
        shape_feature = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), edgecolor="black")
        ax.add_feature(shape_feature, facecolor="none")
        ax.add_feature(cartopy.feature.STATES, edgecolor="black")
        ax.set_extent(
            (hsv_ctr_lon - 2.5, hsv_ctr_lon + 2.5, hsv_ctr_lat - 3.0, hsv_ctr_lat + 2.5)
        )

        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

    # Gridded background
    grid_extent = (ncgrid.x.min(), ncgrid.x.max(), ncgrid.y.min(), ncgrid.y.max())

    # fig.suptitle((t_step[0:19] + ' 40 dbz, long tracks, ISO_THRESH = 12'), fontsize = 12,y=0.76)

    # Cell ID
    im = axs[0].imshow(
        refl,
        origin="lower",
        vmin=-25,
        vmax=85,
        cmap="pyart_LangRainbow12",
        extent=grid_extent,
        transform=grid_proj,
    )
    axs[0].set_title((t_step[0:19]))
    axs.cbar_axes[0].colorbar(im)

    i = np.where(xrdata["segmentation_mask"][t_index, :, :] > 0)
    y1, x1 = (
        ncgrid["point_longitude"].data[0, i[0], i[1]],
        ncgrid["point_latitude"].data[0, i[0], i[1]],
    )  

    axs[0].scatter(y1, x1, s=1, c="gray", marker=".", alpha=0.1, transform=latlon_proj)

    for i in xrdata["cell"]:
        if i < 0:
            continue

        # print(i)
        if math.isfinite(i):
            track_i = np.where(xrdata["feature_parent_cell_id"] == i)
            if (np.nanmax(xrdata["feature_time_index"][track_i]) >= t_index) and (
                np.nanmin(xrdata["feature_time_index"][track_i]) <= t_index
            ):
                axs[0].plot(
                    ncgrid["point_longitude"].data[
                        0,
                        np.round(
                            xrdata["feature_hdim1_coordinate"].data[track_i]
                        ).astype(int),
                        np.round(
                            xrdata["feature_hdim2_coordinate"].data[track_i]
                        ).astype(int),
                    ],
                    ncgrid["point_latitude"].data[
                        0,
                        np.round(
                            xrdata["feature_hdim1_coordinate"].data[track_i]
                        ).astype(int),
                        np.round(
                            xrdata["feature_hdim2_coordinate"].data[track_i]
                        ).astype(int),
                    ],
                    "-.",
                    color="r",
                    markersize=1,
                    transform=latlon_proj,
                )
                axs[0].text(
                    ncgrid["point_longitude"].data[
                        0,
                        np.round(
                            both_ds["feature_hdim1_coordinate"].data[track_i][-1]
                        ).astype(int),
                        np.round(
                            both_ds["feature_hdim2_coordinate"].data[track_i][-1]
                        ).astype(int),
                    ],
                    ncgrid["point_latitude"].data[
                        0,
                        np.round(
                            both_ds["feature_hdim1_coordinate"].data[track_i][-1]
                        ).astype(int),
                        np.round(
                            both_ds["feature_hdim2_coordinate"].data[track_i][-1]
                        ).astype(int),
                    ],
                    f"{int(i)}",
                    fontsize="medium",
                    rotation="vertical",
                    transform=latlon_proj,
                )
        else:
            continue

    for i in xrdata['track']:
        track_i = np.where(xrdata['cell_parent_track_id'] == i.values)
        for cell in xrdata['cell'][track_i]:
            if cell < 0:
                continue
    
            feature_id = np.where(xrdata['feature_parent_cell_id'] == cell)
            if (np.nanmax(xrdata['feature_time_index'][feature_id]) >= t_index) and (np.nanmin(xrdata['feature_time_index'][feature_id]) <= t_index):
                axs[0].plot(ncgrid['point_longitude'].data[0,np.round(xrdata['feature_hdim1_coordinate'].data[feature_id]).astype(int),np.round(xrdata['feature_hdim2_coordinate'].data[feature_id]).astype(int)],
                            ncgrid['point_latitude'].data[0,np.round(xrdata['feature_hdim1_coordinate'].data[feature_id]).astype(int),np.round(xrdata['feature_hdim2_coordinate'].data[feature_id]).astype(int)],
                            '-.',color = 'b',markersize = 1,transform = latlon_proj)
    
                axs[0].text(ncgrid['point_longitude'].data[0,np.round(both_ds['feature_hdim1_coordinate'].data[feature_id][-1]).astype(int),np.round(both_ds['feature_hdim2_coordinate'].data[feature_id][-1]).astype(int)],
                            ncgrid['point_latitude'].data[0,np.round(both_ds['feature_hdim1_coordinate'].data[feature_id][-1]).astype(int),np.round(both_ds['feature_hdim2_coordinate'].data[feature_id][-1]).astype(int)],
                            f'{int(i)}', fontsize = 'small',rotation = 'vertical',transform = latlon_proj)
            else:
                continue

    return


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()


    both_ds = xr.open_dataset(os.path.join(args.tobacpath, "Track_features_merges.nc"))

    plot_dir = args.tobacpath+"/tobac_Plot/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
 
    if args.data_type == 'NEXRAD':
        print('plotting')
        
        data = xr.open_mfdataset(args.path+"*.nc", engine="netcdf4")
        data['time'].encoding['units']="seconds since 2000-01-01 00:00:00"
        bad_rhv = data["cross_correlation_ratio"] < 0.9
        bad_refl = data["reflectivity"] < 10
        bad=bad_rhv & bad_refl
        maxrefl = data["reflectivity"].where(~bad, np.nan).max(axis=1)
        ts = pd.to_datetime(data['time'][0].values)
        date = ts.strftime('%Y%m%d')

        date = args.path[-9:-1]
        
        
        nc_grid = load_cfradial_grids(args.path+"*.nc")
        for i in range(len(nc_grid.time)):
            time_index = i
            fig = plt.figure(figsize=(9, 9))
            fig.set_canvas(plt.gcf().canvas)
            plot(time_index, both_ds, maxrefl, nc_grid,args.plot_lat, args.plot_lon)
            fig.savefig(plot_dir + date+"_tobac_"+ str(time_index) +".png")
            plt.close(fig)

    if args.data_type == 'NUWRF':
    
        files = sorted(glob(args.path+"wrfout_d01_2022-08-07_18*00"))
        data1 = xr.open_dataset(files[0])
        drop_list = list(np.sort(list(data1.variables)))
        drop_list = [e for e in drop_list if e not in ('COMDBZ', 'Times','XLAT','XLONG','XTIME')]



        import xwrf
        data = xr.open_mfdataset(files, engine="netcdf4",parallel=True,
            concat_dim="Time", combine="nested", chunks={"Time": 1},decode_times=False,
            drop_variables=drop_list,).xwrf.postprocess()


        #MAKE THE TIME DIMENSION AND COORDINATES PLAY NICE
        data = data.rename_dims({'Time': 'time'})
        data['time'] = data['Time']
        maxrefl = data['COMDBZ']
        maxrefl = maxrefl.drop('XTIME')
        maxrefl = maxrefl.drop('Time')


        ref_levels = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
        for j in range(len(maxrefl.time)):
            date = str(maxrefl['time'][j].values)[:-13]
            time_index = j
            fig, ax = plt.subplots(figsize=(10,10))
            refl = maxrefl[j,:,:].values
            fig.suptitle(str(maxrefl['time'][j].data)[:-10])

            y_mesh = maxrefl['XLAT'].values
            x_mesh = maxrefl['XLONG'].values
            refplt = ax.contourf(x_mesh,y_mesh, refl, extend = 'max',levels = ref_levels,cmap='pyart_LangRainbow12',origin = 'lower',
            vmin=-24, vmax=72, extent = [-96,-93,28,30])

            fig.colorbar(refplt,fraction=0.046, pad=0.04)
            i = np.where(both_ds['segmentation_mask'][j,:,:] > 0)
            y, x = y_mesh[i[0],i[1]],x_mesh[i[0],i[1]]
            imcell2 = ax.scatter(x,y,s = 0.5,c = 'gray', marker = '.',alpha = 0.75)

            for i in both_ds['track']:
                track_i = np.where(both_ds['cell_parent_track_id'] == i.values)
                for cell in both_ds['cell'][track_i]:
                    if cell < 0:
                        continue

                    feature_id = np.where(both_ds['feature_parent_cell_id'] == cell)
                    if (j <= np.nanmax(both_ds['feature_time_index'][feature_id])) and (j >= np.nanmin(both_ds['feature_time_index'][feature_id])):
                        ax.plot(both_ds['wrf_XLONG'][feature_id], both_ds['wrf_XLAT'][feature_id], '-.',color='b',alpha = 0.5)
                        ax.text(both_ds['wrf_XLONG'][feature_id][-1],both_ds['wrf_XLAT'][feature_id][-1], f'{int(i)}', fontsize = 'small',rotation = 'vertical')
                    else:
                        continue

                fig.savefig(plot_dir + date+"_tobac_NUWRF_"+str(time_index) + ".png")
                plt.close(fig)
                
                
    if args.data_type == 'POLARRIS':


        data = xr.open_mfdataset(args.path+'*.nc', engine = 'netcdf4',combine = 'nested' ,concat_dim='time')
        data['time'].encoding['units']="seconds since 2000-01-01 00:00:00"
        files = sorted(glob(args.path+'*.nc'))
        arr = []
        for i in files:
            arr.append(pd.to_datetime(i[-19:-3], format = '%Y_%m%d_%H%M%S'))
        arr = pd.DatetimeIndex(arr)
        data = data.assign_coords(time=arr)


        bad_rhv = data["RH"] < 0.9
        bad_refl = data["CZ"] < 10
        bad=bad_rhv & bad_refl
        maxrefl = data["CZ"].where(~bad, np.nan).max(axis=1)

        nc_grid = load_cfradial_grids_polarris(data)
        for i in range(len(nc_grid.time)):
            time_index = i
            fig = plt.figure(figsize=(9,9))
            fig.set_canvas(plt.gcf().canvas)
            plot(time_index,both_ds,maxrefl,nc_grid,args.plot_lat, args.plot_lon)
            fig.savefig(plot_dir + date+"_tobac_NUWRF_"+str(time_index)+".png")
            plt.close(fig)

    if args.data_type == 'POLARRIS2':
        import numpy.ma as ma
        data = xr.open_mfdataset(args.path + "*POLARRIS.matsui2018.nc", engine="netcdf4", combine="nested", concat_dim="time")
        data["time"].encoding["units"] = "seconds since 2000-01-01 00:00:00"
        files = sorted(glob(args.path + "*POLARRIS.matsui2018.nc"))
        arr = []
        for i in files:
            arr.append(pd.to_datetime(i[-42:-23], format="%Y-%m-%d_%H:%M:%S"))
        arr = pd.DatetimeIndex(arr)
        data = data.assign_coords(time=arr)

        bad_rhv = data["rhohv01"] < 0.9
        bad_refl = data["zhh01"] < 10
        bad = bad_rhv & bad_refl
        maxrefl = data["zhh01"].where(~bad, np.nan).max(axis=1)

    # Dt, DXY
        datetimes = data["time"]
        timedeltas = [
        (datetimes[i - 1] - datetimes[i]).astype("timedelta64[m]")
        for i in range(1, len(datetimes))]
        average_timedelta = sum(timedeltas) / len(timedeltas)
        dt = np.abs(np.array(average_timedelta)).astype("timedelta64[m]").astype(int)
    
        dxy = args.dxy

        ts = pd.to_datetime(data["time"][0].values)
        date = ts.strftime("%Y%m%d")

        ref_levels = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
        for j in range(len(maxrefl.time)):
            fig, ax = plt.subplots(figsize=(10,10))
            date = str(maxrefl['time'][j].values)[:-13]
            time_index = j
            refl = maxrefl[j,:,:].values
            fig.suptitle(str(maxrefl['time'][j].data)[:-10])
            refplt = ax.contourf(data['longitude'][j,:,:],data['latitude'][j,:,:], refl, extend = 'max',levels =                          ref_levels,cmap='pyart_LangRainbow12',origin = 'lower',  vmin=-24, vmax=72, extent = [-96,-93,28,30])
            fig.colorbar(refplt,fraction=0.046, pad=0.04)
            mask  = ma.masked_where(both_ds['segmentation_mask'][j,:,:] <=0,both_ds['segmentation_mask'][j,:,:])
            imcell2 = ax.contourf(data['longitude'][j,:,:],data['latitude'][j,:,:],mask,cmap = 'gray',alpha = 0.6)
            
        
            for i in both_ds['track']:
                track_i = np.where(both_ds['cell_parent_track_id'] == i.values)
                for cell in both_ds['cell'][track_i]:
                    if cell < 0:
                        continue

                    feature_id = np.where(both_ds['feature_parent_cell_id'] == cell)
                    if (j <= np.nanmax(both_ds['feature_time_index'][feature_id])) and (j >= np.nanmin(both_ds['feature_time_index'][feature_id])):
                

                        ax.plot(data['longitude'][j,np.round(both_ds['feature_hdim1_coordinate'][feature_id]).astype(int),np.round(both_ds['feature_hdim2_coordinate'][feature_id]).astype(int)],
                        data['latitude'][j,np.round(both_ds['feature_hdim1_coordinate'][feature_id]).astype(int),np.round(both_ds['feature_hdim2_coordinate'][feature_id]).astype(int)], '-.',color='b',alpha = 0.5)
    
                        ax.text(data['longitude'][j,np.round(both_ds['feature_hdim1_coordinate'][feature_id]).astype(int),np.round(both_ds['feature_hdim2_coordinate'][feature_id]).astype(int)][-1],
                        data['latitude'][j,np.round(both_ds['feature_hdim1_coordinate'][feature_id]).astype(int),np.round(both_ds['feature_hdim2_coordinate'][feature_id]).astype(int)][-1], f'{int(i)}', fontsize = 'small',rotation = 'vertical')

                    else:
                        continue

            fig.savefig(plot_dir + date+"_tobac_POLARRIS_"+str(time_index) + ".png")
            plt.close(fig)
         