#!/usr/bin/env python
# coding: utf-8

import argparse

parse_desc = """ Create histograms of track properties from tobac tracks and the associated
timeseries data along the tracks. Also requires one reference gridded radar file for the projection information.



The path is a string destination
Files in path must have a postfix of '.nc', this will be searched for internally.
Three paths are required: 
path: path to the radar data, currently only NEXRAD data is supported in this version
lmapath: path to the lma flash sorted data
tobacpath: path to the tobac feature, track etc. netcdf files.
type: Name of the type of data (NEXRAD/POLARRIS/NUWRF) given as all uppercase string. Currently only NEXRAD is supported.


Example
=======
python tobac_track_agg.py --main-path='/Volumes/LtgSSD/tobac_saves/' \
                                 --referencegridpath='/efs/tracer/NEXRAD/20220604/KHGX20220604_000224_V06_grid.nc' \
                                 --distance=90.0
"""

from glob import glob
import os


def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(
        "--referencegridpath",
        metavar="referencegridpath",
        required=True,
        dest="referencegridpath",
        action="store",
        help="path to a sample radar grid dataset containing projection information for the feature mask in the track dataset",
    )
    parser.add_argument(
        "--main-path",
        required=True,
        action="store",
        help="tobac_Save subdir",
    )
    parser.add_argument(
        "--distance",
        metavar="KHGX distance",
        default = 90.0,
        dest="khgx_distance_km",
        action="store",
        help="Maximum track distance from KHGX, in km. At least one feature in the track must be within range.",
        type=float
    )
    return parser


# End parsing #

# Import packages
import os
import warnings
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import KDTree

from pyxlma.lmalib.traversal import OneToManyTraversal

try:
    import pyproj

    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

    
csapr_lon, csapr_lat = -95.283893, 29.531782
khgx_lat, khgx_lon = 29.4719, -95.0788


def feature_distance_from(ds, lon, lat, label):
    """ Calculate x, y, and total distances of each feature from (lon, lat) and add them
    to ds as a new variable named like 'feature_label_x_dist', 'feature_label_y_dist'
    and 'feature_label_dist'.
    """
    dlon, dlat = ds.grid_longitude-lon, ds.grid_latitude-lat
    dlonlatsq = dlon*dlon+dlat*dlat
    y_idx, x_idx = np.unravel_index(np.argmin(dlonlatsq.values), dlonlatsq.shape)
    x, y = ds.y[y_idx], ds.x[x_idx]
    name = 'feature_'+label
    ds[name+'_x_dist'] = ds.feature_projection_x_coordinate - x
    ds[name+'_y_dist'] = ds.feature_projection_y_coordinate - y
    ds[name+'_dist'] = (ds[name+'_x_dist']*ds[name+'_x_dist'] + 
                        ds[name+'_y_dist']*ds[name+'_y_dist'])**0.5
    return ds

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


# Count neighbors
def count_track_neighbors(track_ds, distance_thresholds = (5.0, 10.0, 15.0, 20.0), grid_spacing = 0.5):
    feature_neighbor_variable_names = []

    # First find the trees corresponding to all features at each time.
    time_groups = track_ds.groupby('feature_time_index')
    time_groups.groups.keys()
    trees_each_time_index = {}
    for time_idx, group in time_groups:
        hdim1 = group['feature_hdim1_coordinate'].values*grid_spacing
        hdim2 = group['feature_hdim2_coordinate'].values*grid_spacing
        #note hdim1,2 are in km
        pts = np.vstack((hdim2, hdim1)).T
        tree = KDTree(pts)
        trees_each_time_index[time_idx] = tree

    # Now we'll look at each feature in turn, and its neighbors at that time.
    hdim1 = track_ds['feature_hdim1_coordinate'].values*grid_spacing
    hdim2 = track_ds['feature_hdim2_coordinate'].values*grid_spacing
    pts = np.vstack((hdim2, hdim1)).T
    #note hdim1,2 are in km
    for distance_threshold in distance_thresholds:
        num_obj = np.zeros(len(track_ds["feature"].values), dtype=int)
        for i, ind in enumerate(track_ds["feature"].values):
            time_idx = track_ds.feature_time_index.values[i]
            tree = trees_each_time_index[time_idx]
            # Need to subtract one, since the feature itself is always near (at) the test location
            num_obj[i]=len(tree.query_ball_point(pts[i],r=distance_threshold)) - 1 
        this_nearby_var_name = 'feature_nearby_count_{0}km'.format(int(distance_threshold))
        feature_neighbor_variable_names.append(this_nearby_var_name)
        track_ds = track_ds.assign(**{this_nearby_var_name:(['feature'], num_obj)})
    return track_ds

def add_track_durations(combo):
    features_by_track = combo.drop_dims(('x','y','time')).groupby('feature_parent_track_id')
    maxxed_features = features_by_track.max(dim='feature')
    minned_features = features_by_track.min(dim='feature')

    total_track_duration = maxxed_features.feature_time - minned_features.feature_time
    # feature_parent_track_id becomes the dimension of the array after the groupby, and that coordinate
    # variable contains the track IDs. Rename that dimension to track, and then assign it back to the original
    # combo dataset to store the track duration.
    combo['track_duration']=total_track_duration.rename(feature_parent_track_id='track')

    # Now replicate the track durations down to the feature level
    the_dims = getattr(combo, 'feature_parent_track_id').dims
    # These are replicated exactly as we wish - the feature_parent_track_id is necessarily repeated.
    replicated_track_ids = getattr(combo, 'feature_parent_track_id')
    # When we .loc along the track dimensino with feature_parent_track_id, the values are repeated.
    replicated_data = combo.loc[{'track':replicated_track_ids}]
    # Since the values are repeated in the same order as feature_parent_track_id, we 
    # can get the raw values array and assign it back to the feature dimension.
    combo['feature_parent_track_duration'] = (the_dims, replicated_data.track_duration.values)
    return combo

def track_polarimetry(summed_features,    
        zdr_thresh = 0.0,
        kdp_thresh = 0.0,
        flash_thresh = 0.0,):
    """
    Calculate the polarimetric column and lightning flash properties of a tracked dataset.
    summed_features: a tracked dataset that has been grouped by feature_parent_track_id, 
                     and then summed over the feature dimension.
    
    Returns (track_membership, counts), a dictionary of DataArrays giving boolean membership of each 
    track in each category, and a pandas DataFrame that is a count of tracks in those categories.
    """

    has_zdr = (summed_features.feature_zdrvol > zdr_thresh)
    no_zdr = ~has_zdr
    has_kdp = (summed_features.feature_kdpvol > kdp_thresh)
    no_kdp = ~has_kdp
    has_lightning = (summed_features.feature_flash_count > flash_thresh)
    no_lightning = ~has_lightning

    track_membership = dict(
        track_has_zdr_kdp_ltg = (has_kdp & has_zdr & has_lightning),
        track_has_zdr_kdp_only = (has_kdp & has_zdr & no_lightning),
        track_has_zdr_ltg_only = (no_kdp & has_zdr & has_lightning),
        track_has_zdr_only = (no_kdp & has_zdr & no_lightning),
        track_has_nothing = (no_kdp & no_zdr & no_lightning),
        track_has_kdp_only = (has_kdp & no_zdr & no_lightning),
        track_has_kdp_ltg_only = (has_kdp & no_zdr & has_lightning),
        track_has_ltg_only = (no_kdp & no_zdr & has_lightning),
    )
    
    #header = ["nothing","zdr","kdp","kdp_zdr","ltg","kdp_zdr_ltg","kdp_ltg","zdr_ltg"]
    #results_row = np.fromiter(map(sum, 
    #                          [has_nothing,has_zdr_only,has_kdp_only,has_zdr_kdp_only,
    #                           has_ltg_only,has_zdr_kdp_ltg,has_kdp_ltg_only,has_zdr_ltg_only]),
    #                      dtype=int)
    # counts = pd.DataFrame([results_row,], columns=header)
    results = {k:[v.sum().values] for k,v in track_membership.items()}
    counts = pd.DataFrame(results)
    
    
    return track_membership, counts


def open_track_timeseries_dataset(track_filename, timeseries_filename, reference_grid):
    # =======Load the tracks======
    
    track_ds = xr.open_dataset(track_filename, decode_timedelta=False)

    # In the feature calculation script, the entity IDs need to be ints.
    for var in ['track', 'cell_parent_track_id', 'feature_parent_track_id', 'track_child_cell_count', 'cell_child_feature_count']:
        track_ds[var] = track_ds[var].astype('int64')

    # Get a sample radar dataset from this day to get its projection data. The projection varaiables
    # should also be copied into track dataset, but here here we read it from one of the grids, and
    # use it to calculate the longitude/latitude data, too.
    refl = xr.open_dataset(reference_grid, decode_timedelta=False)
    lon, lat = cartesian_to_geographic(refl)
    track_ds['grid_longitude'] = xr.DataArray(lon[0,:,:], dims=('y', 'x'))
    track_ds['grid_latitude'] = xr.DataArray(lat[0,:,:], dims=('y', 'x'))
    track_ds['projection'] = refl.projection
    track_ds['ProjectionCoordinateSystem'] = refl.ProjectionCoordinateSystem 

    track_ds = count_track_neighbors(track_ds, grid_spacing=0.5)
    
    # ======Load the timeseries=======
        
    ds = xr.open_dataset(timeseries_filename)
    
    # ======Combine both datasets and calculate some generally useful properties=======
    combo = xr.combine_by_coords((ds, track_ds))

    # Index location of each feature; feature coordiantes are averages, so we need to convert to ints.
    hdim1_i = combo.feature_hdim1_coordinate.astype('int64')
    hdim2_i = combo.feature_hdim2_coordinate.astype('int64')

    combo['feature_longitude_center'] = combo['grid_longitude'][hdim1_i, hdim2_i]
    combo['feature_latitude_center'] = combo['grid_latitude'][hdim1_i, hdim2_i]


    # Some variables have negative values that indicate 
    return combo


def main(args):
    main_path = args.main_path
    track_paths = sorted(glob(f'{main_path}/tobac_Save_*/Track_features_merges.nc'))
    track_dates = [datetime.strptime(os.path.dirname(tp).split('_')[-1], '%Y%m%d') for tp in track_paths]
    timeseries_paths = sorted(glob(f'{main_path}/tobac_Save_*/timeseries_data_melt*.nc'))
    timeseries_dates = [datetime.strptime(os.path.dirname(tp).split('_')[-1].replace('.nc',''), '%Y%m%d') for tp in timeseries_paths]
    both_dates = sorted(list(set(track_dates).intersection(set(timeseries_dates))))
    track_paths = [tp for tp, d in zip(track_paths, track_dates) if d in both_dates]
    timeseries_paths = [tp for tp, d in zip(timeseries_paths, timeseries_dates) if d in both_dates]
    print(f'Processing {len(both_dates)} days with both track and timeseries data.')
    print(', '.join([d.strftime('%Y-%m-%d') for d in both_dates]))
    all_tracks = pd.DataFrame()
    for i, (trackpath, timeseriespath) in enumerate(zip(track_paths, timeseries_paths)):
        print(f'Processing day: {both_dates[i].strftime("%Y-%m-%d")}')
        combo = open_track_timeseries_dataset(
            trackpath, timeseriespath, reference_grid=args.referencegridpath)
        
        # These first steps could be broken into their own script for preprocessing the timeseries dataset
        # to contain a certain subset of the tracks. The later sections are a uniform 
        combo = feature_distance_from(combo, csapr_lon, csapr_lat, 'csapr')
        combo = feature_distance_from(combo, khgx_lon, khgx_lat, 'khgx')
        
        # Set up a prunable tree for the track,cell,feature dataset, replacing the old one now that combo has all vars
        traversal = OneToManyTraversal(combo, ('track','cell','feature'), ('cell_parent_track_id', 'feature_parent_cell_id'))

        # Find tracks IDs with at least one feature within the specified range
        feature_in_range = (combo['feature_khgx_dist'] < (args.khgx_distance_km * 1000.0))
        reduced_track_ids = np.unique(combo[{'feature':feature_in_range}].feature_parent_track_id)
        combo = traversal.reduce_to_entities('track', reduced_track_ids)

        combo = add_track_durations(combo)

        these_tracks = np.arange(len(all_tracks), len(all_tracks)+combo.track.size)
        track_area_sum = np.zeros(len(these_tracks))
        track_area_avg = np.zeros(len(these_tracks))
        these_kdp_sum = np.zeros(len(these_tracks))
        these_kdp_avg = np.zeros(len(these_tracks))
        these_zdr_sum = np.zeros(len(these_tracks))
        these_zdr_avg = np.zeros(len(these_tracks))
        these_ltg = np.zeros(len(these_tracks))

        for j in range(len(these_tracks)):
            track_id = combo.isel(track=j).track.data
            feature_mask = (combo.feature_parent_track_id.data == track_id)
            track_features = combo.isel(feature=feature_mask)
            track_area_sum[j] = track_features.feature_area.sum().values
            track_area_avg[j] = track_features.feature_area.mean().values
            these_kdp_sum[j] = track_features.feature_kdpwt_total.sum().values
            these_zdr_sum[j] = track_features.feature_zdrwt_total.sum().values
            these_kdp_avg[j] = track_features.feature_kdpwt_total.mean().values
            these_zdr_avg[j] = track_features.feature_zdrwt_total.mean().values
            these_ltg[j] = track_features.feature_flash_count.sum().values
        these_tracks_df = pd.DataFrame(dict(
            track_id=combo.track.values,
            track_duration=combo.track_duration.data.astype('timedelta64[s]').astype(float),
            track_kdpwt_sum=these_kdp_sum,
            track_kdpwt_avg=these_kdp_avg,
            track_zdrwt_sum=these_zdr_sum,
            track_zdrwt_avg=these_zdr_avg,
            track_area_sum=track_area_sum,
            track_area_avg=track_area_avg,
            track_flash_count=these_ltg
        ))
        these_tracks_df['track_date'] = both_dates[i].strftime('%Y-%m-%d')
        all_tracks = pd.concat((all_tracks, these_tracks_df))

    output_path = os.path.join(main_path, 'all_tracks.csv')
    all_tracks.to_csv(output_path, index=False)
    print(f'Wrote aggregated track data to {output_path}')

    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    xr.set_options(file_cache_maxsize=1)
    main(args)
