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
python tobac_track_histograms.py --trackpath='/efs/tracer/NEXRAD/tobac_Save_20220604/Track_features_merges.nc' \
                                 --timeseriespath='/efs/tracer/NEXRAD/tobac_Save_20220604/timeseries_data_melt4400.nc' \
                                 --referencegridpath='/efs/tracer/NEXRAD/20220604/KHGX20220604_000224_V06_grid.nc' \
                                 --output_path='/efs/tracer/NEXRAD/tobac_Save_20220604/' \
                                 --distance=150.0
"""

import linecache
import os
import tracemalloc


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
        "--timeseriespath",
        metavar="timeseriespath",
        required=True,
        dest="timeseriespath",
        action="store",
        help="path to the timeseries dataset containing the polarimetry and lightning data along each tobac track",
    )
    parser.add_argument(
        "--trackpath",
        metavar="trackpath",
        required=True,
        dest="trackpath",
        action="store",
        help="path to the tobac track dataset",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        metavar="output path",
        required=False,
        dest="outdir",
        action="store",
        default=".",
        help="path where the data will be saved",
    )
    parser.add_argument(
        "--distance",
        metavar="KHGX distance",
        default = 150.0,
        dest="khgx_distance_km",
        action="store",
        help="Maximum track distance from KHGX, in km. At least one feature in the track must be within range.",
        type=float
    )


    # parser.add_argument(
    #     "--type",
    #     metavar="data type",
    #     required=False,
    #     dest="data_type",
    #     action="store",
    #     help="Data name type, e.g., NEXRAD, POLARRIS, NUWRF",
    # )
    return parser


# End parsing #

# Import packages
import os
import warnings
from datetime import datetime
from functools import partial
from itertools import combinations
import random

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import KDTree

from xhistogram.xarray import histogram
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
    from scipy.spatial import KDTree

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


def open_track_timeseries_dataset(track_filename, timeseries_filename, reference_grid=None):
    # =======Load the tracks======
    
    track_ds = xr.open_dataset(track_filename)

    # In the feature calculation script, the entity IDs need to be ints.
    for var in ['track', 'cell_parent_track_id', 'feature_parent_track_id', 'track_child_cell_count', 'cell_child_feature_count']:
        track_ds[var] = track_ds[var].astype('int64')

    # Get a sample radar dataset from this day to get its projection data. The projection varaiables
    # should also be copied into track dataset, but here here we read it from one of the grids, and
    # use it to calculate the longitude/latitude data, too.
    refl = xr.open_dataset(reference_grid)
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

    # dim 0 is time. 1 is usually y, 2 is usually x
    hdim1_var = combo.segmentation_mask.dims[1]
    hdim2_var = combo.segmentation_mask.dims[2]

    # Index location of each feature; feature coordiantes are averages, so we need to convert to ints.
    hdim1_i = combo.feature_hdim1_coordinate.astype('int64')
    hdim2_i = combo.feature_hdim2_coordinate.astype('int64')

    combo['feature_longitude_center'] = combo['grid_longitude'][hdim1_i, hdim2_i]
    combo['feature_latitude_center'] = combo['grid_latitude'][hdim1_i, hdim2_i]


    # Some variables have negative values that indicate 
    return combo


def subdivide_tracks(ds, track_membership):
    sub_dss = {}
    
    # Set up a prunable tree for the track,cell,feature dataset
    traversal = OneToManyTraversal(ds, 
        ('track','cell','feature'), 
        ('cell_parent_track_id', 'feature_parent_cell_id'))
        
    all_features_by_track = ds.drop_dims(('x','y','time')).groupby('feature_parent_track_id')
    all_summed_features = all_features_by_track.sum(dim='feature')
    
    sub_dss['track_has_any'] = ds, all_summed_features
    
    for kind, membership in track_membership.items():
        reduced_track_ids = membership.where(membership, drop=True).feature_parent_track_id.values
        # ds.track[membership].values
        sub = traversal.reduce_to_entities('track', reduced_track_ids)
        
        # Need to recalculate features_by_track now that we've added all variables to combo.
        if sub.feature_parent_track_id.shape[0] > 0:
            features_by_track = sub.drop_dims(('x','y','time')).groupby('feature_parent_track_id')
            summed_features = features_by_track.sum(dim='feature')
        else:
            summed_features = None
        
        sub_dss[kind] = sub, summed_features
        
    return sub_dss
    
    
def percents_and_histos(summed_features, var_bins, percentiles):
   
    percentiles_out = percentiles.copy()
    
    raw_var_histos = {}
    for var, (description, bins) in var_bins.items():
        if summed_features is not None:
            data = summed_features[var]
            non_zero = (data > 0) & np.isfinite(data)
            counts, bins = np.histogram(data, bins=bins)    
            if non_zero.sum()>0:
                percentiles_out[var] = np.percentile(data[non_zero], percentiles['thresholds'])
                raw_var_histos[var] = counts
            else:
                percentiles_out[var] = np.nan * np.asarray(percentiles['thresholds'])
                raw_var_histos[var] = np.zeros((bins.shape[0]-1,), dtype=int)
        else:
            percentiles_out[var] = np.nan * np.asarray(percentiles['thresholds'])
            raw_var_histos[var] = np.zeros((bins.shape[0]-1,), dtype=int)
        print(var, percentiles_out[var])

    all_joint_var_combos = list(combinations(var_bins.keys(),2))

    # normed_joint_var_combos = [(var1, var2) for (var1, var2) in all_joint_var_combos if ( ('norm' in var1) & ('norm' in var2) )]
    # total_plot_combinations = len(all_joint_var_combos)
    # normed_plot_combinations = len(normed_joint_var_combos)

    joint_var_combos = all_joint_var_combos

    all_histos = []
    for axi, (var1, var2) in enumerate(joint_var_combos):
        if (not ('norm' in var1)) & (not ('norm' in var2)):
            continue
        bins = [var_bins[var1][1], var_bins[var2][1]]
        if summed_features is not None:
            h = histogram(summed_features[var1],
                          summed_features[var2], 
                          bins=bins)
            all_histos.append(h.compute())
        else:
            # Fake an empty histogram. We confirm this is the right structure to mimic the expected
            # variable naming by using the following code to see what xhistogram does:
            # """
            # from xhistogram.xarray import histogram
            # import xarray as xr
            # import numpy as np

            # foo_bin = np.linspace(-4, 4, 20)
            # bar_bin = foo_bin -2
            # bins=[foo_bin, bar_bin]

            # nt, nx = 100, 30
            # da = xr.DataArray(np.random.randn(nt, nx), dims=['time', 'x'],
            #                   name='foo') # all inputs need a name
            # db = xr.DataArray(np.random.randn(nt, nx), dims=['time', 'x'],
            #                   name='bar') - 2

            # h = histogram(da, db, bins=bins)
            # print(h) # or display h in notebook
            # """
            zeros = np.zeros((bins[0].shape[0]-1, bins[1].shape[0]-1,), dtype=int)
            bin_centers = [0.5 * (bin[:-1] + bin[1:]) for bin in bins] # Ripped straight from xhistogram
            h = xr.DataArray(zeros,  coords={var1+'_bin':bin_centers[0], var2+'_bin':bin_centers[1]}, 
                             dims=(var1+'_bin', var2+'_bin'), 
                             name='_'.join(['histogram', var1, var2]))
            all_histos.append(h.compute())
    histo_ds = xr.combine_by_coords(all_histos)
    
    return raw_var_histos, histo_ds, percentiles_out


def main(args):

    combo = open_track_timeseries_dataset(
        args.trackpath, args.timeseriespath, reference_grid=args.referencegridpath)
    
    meltlevel_string = args.timeseriespath.split('_')[-1].replace('.nc', '')
    
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

    # Normalization is by feature area at each time and then by the total duration of the track.
    vars_to_normalize = ['feature_flash_count', 
                         # when normalized ... will be the flash rate per unit area per unit time of the whole track.

                         'feature_zdrvol', 'feature_kdpvol', 'feature_rhvdeficitvol', 
                          # when normalized ... will be an average column depth, normalized by track duration.

                         'feature_zdrcol', 'feature_kdpcol', 'feature_rhvdeficitcol',
                          # ... will be a max divided by an area, normalized by track duration.. Larger areas will make strong Kdp values stand out less, 
                          # which is fine for our purposes, where we want to highlight relatively small, isolated cells.

                         'feature_zdrcol_total', 'feature_kdpcol_total', 'feature_rhvdeficitcol_total',
                          # ... will be area average of the integral along the vertical dimension, normalized by track duration, so proportional to feature_*_mean.
                         'feature_zdrwt_total', 'feature_kdpwt_total', 'feature_rhvdeficitwt_total',

                         'feature_zdrcol_mean', 'feature_kdpcol_mean', 'feature_rhvdeficitcol_mean',
                          # ... will be the average value along the vertical dimension, per unit area,normalized by track duration.

                         'feature_nearby_count_20km',
                          # ... will be the average neighbors per unit area per unit time.
                        ]

    # Summing means that long-lived and large-sized things can look the same.
    # We want to take each of these properties, and then normalize by their area.
    # We also want to normalize by duration.
    track_duration_sec = combo.feature_parent_track_duration.values.astype('timedelta64[s]').astype(float)

    for var in vars_to_normalize:
        combo[var+'_area_time_norm'] = combo[var]/combo.feature_area/track_duration_sec


    ### Sum properties and prepare to look at distributions

    # Need to recalculate features_by_track now that we've added all variables to combo.
    features_by_track = combo.drop_dims(('x','y','time')).groupby('feature_parent_track_id')
    summed_features = features_by_track.sum(dim='feature')
    track_membership, track_counts = track_polarimetry(summed_features, zdr_thresh=0.0, kdp_thresh=0.0, flash_thresh=0.0)
    
    subdivided_tracks = subdivide_tracks(combo, track_membership)

    ### Start calculating stats.
    
    pow2 = partial(pow, 2)
    powers_two = np.array([-1, 0] + list(map(pow2, range(20))) )+0.5

    # Values from the 4 June case, rounded to 40 x 40 km and one hour.
    mean_area = 1600.0 # combo.feature_area.mean().values
    mean_duration_sec = 3600.0 # combo.track_duration.mean().values.astype('timedelta64[s]').astype(float)

    # for the altitude-weighted variables, also normalize by the maximum magnitude of a weight.
    alt_weight_max = 20000
    
    var_bins= dict(
        feature_flash_count = ('Flash count', 
                               powers_two),
        feature_zdrvol = (r'Track-total $Z_{DR}$ column volume (grid box count)',
                          powers_two),
        feature_kdpcol = (r'Track-total $K_{DP}$ column volume (grid box count)',
                          powers_two),
        feature_rhvdeficitcol = (r'Track-total $rho_{hv}$ deficit column volume (grid box count)',
                          powers_two),

        feature_flash_count_area_time_norm = (r'Flash count per feature area normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),
        
        feature_zdrvol_area_time_norm = (r'Sum along track of $Z_{DR}$ average column depth at each time'
                                          '\n(grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),
        feature_kdpvol_area_time_norm = (r'Sum along track of $K_{DP}$ average column depth at each time'
                                         '\n(grid box count), normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),
        feature_rhvdeficitvol_area_time_norm = (r'Sum along track of $rho_{hv}$ deficit average column depth at each time'
                                         '\n(grid box count), normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),

        feature_zdrcol_area_time_norm = (r'Sum along track of max(integrated $Z_{DR}$ along one grid box column)'
                                          '\nin each feature (grid box count),normalized by track area and duration',
                                    powers_two / mean_area / mean_duration_sec),
        feature_kdpcol_area_time_norm = (r'Sum along track of max(integrated $K_{DP}$ along one grid box column)'
                                          '\nin each feature (grid box count),normalized by track area and duration',
                                    powers_two / mean_area / mean_duration_sec),
        feature_rhvdeficitcol_area_time_norm = (r'Sum along track of max(integrated $rho_{hv}$ deficit along one grid box column)'
                                          '\nin each feature (grid box count),normalized by track area and duration',
                                    powers_two / mean_area / mean_duration_sec),


        feature_zdrcol_total_area_time_norm = (r'Area average of the integral of $Z_{DR}$ along the vertical dimension'
                                          '\nin each feature (grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),
        feature_kdpcol_total_area_time_norm = (r'Area average of the integral of $K_{DP}$ along the vertical dimension'
                                          '\nin each feature (grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),
        feature_rhvdeficitcol_total_area_time_norm = (r'Area average of the integral of $rho_{hv}$ deficit along the vertical dimension'
                                          '\nin each feature (grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec),
        
        
        feature_zdrwt_total_area_time_norm = (r'Area average of the integral of $Z_{DR}$ along the vertical dimension weighted by altitude'
                                          '\nin each feature (grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec * alt_weight_max),
        feature_kdpwt_total_area_time_norm = (r'Area average of the integral of $K_{DP}$ along the vertical dimension weighted by altitude'
                                          '\nin each feature (grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec * alt_weight_max),
        feature_rhvdeficitwt_total_area_time_norm = (r'Area average of the integral of $rho_{hv}$ deficit along the vertical dimension weighted by altitude'
                                          '\nin each feature (grid box count),normalized by track duration',
                                    powers_two / mean_area / mean_duration_sec * alt_weight_max),

        feature_nearby_count_20km_area_time_norm = (r'Count of all nearby features along track within 20 km,'
                                                    '\nnormalized by feature area and track duration.',
                                    powers_two / mean_area / mean_duration_sec),
    )


    feature_neighbor_variable_names = [v for v in combo.data_vars if (
        ('feature_nearby_count' in v) & (not('norm' in v)))]
    for var in feature_neighbor_variable_names:
        distance = var.split('_')[-1]
        var_bins[var] = ('Count of all nearby features along track within '+distance, powers_two)

    percentiles = dict(thresholds=(5, 50, 95))
    
    # Moved into function
#     raw_var_histos = {}
#     for var, (description, bins) in var_bins.items():
#         data = summed_features[var]
#         non_zero = (data > 0) & np.isfinite(data)
#         counts, bins = np.histogram(data, bins=bins)    
#         percentiles[var] = np.percentile(data[non_zero], percentiles['thresholds'])
#         print(var, percentiles[var])
#         raw_var_histos[var] = counts

#     all_joint_var_combos = list(combinations(var_bins.keys(),2))

#     # normed_joint_var_combos = [(var1, var2) for (var1, var2) in all_joint_var_combos if ( ('norm' in var1) & ('norm' in var2) )]
#     # total_plot_combinations = len(all_joint_var_combos)
#     # normed_plot_combinations = len(normed_joint_var_combos)

#     joint_var_combos = all_joint_var_combos

#     all_histos = []

#     for axi, (var1, var2) in enumerate(joint_var_combos):
#         if (not ('norm' in var1)) & (not ('norm' in var2)):
#             continue
#         bins = [var_bins[var1][1], var_bins[var2][1]]
#         h = histogram(summed_features[var1],
#                       summed_features[var2], 
#                       bins=bins)
#         all_histos.append(h.compute())

#     histo_ds = xr.combine_by_coords(all_histos) 
    # End of move into function
    
    
    for kind, (subset_ds, summed_features_subset) in subdivided_tracks.items():
        print(kind)
        raw_var_histos, histo_ds, percentiles_out = percents_and_histos(summed_features_subset, var_bins, percentiles)
    
        # Add a variable to show time coverage
        histo_ds['grid_time_start'] = combo.time.min()
        histo_ds['grid_time_end'] = combo.time.max()

        for k, var in raw_var_histos.items():
            # We can reuse the bins that were already put in the dataset for the 2D histograms
            bins_coord = k+'_bin'
            histo_ds[k] = xr.DataArray(var, dims=(bins_coord,))
            histo_ds[k].attrs["long_name"] = var_bins[k][0]

        percentile_ds = {'percentile_'+k:{"dims":"percentile_thresholds", "data":np.asarray(v)}
                         for k, v in percentiles_out.items()}
        percentile_ds = xr.Dataset.from_dict(percentile_ds)
        # for var in percentile_ds.data_vars:
        #     bin_key = var.replace('percentile_', '')
        #     percentile_ds[var].attrs["long_name"] = var_bins[bin_key][0]

        # Add the track_counts dataframe, which has only one row, here.
        track_counts_out = track_counts.to_xarray().rename({'index':'track_count'})    

        histo_ds = xr.combine_by_coords((histo_ds, percentile_ds, track_counts_out))
        histo_ds['track_maximum_distance_km'] = args.khgx_distance_km

        histo_ds.to_netcdf(os.path.join(args.outdir, "histogram_data_{1}_{0}.nc".format(meltlevel_string, kind.replace('_','-'))))
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    import dask
    import xarray as xr
    xr.set_options(file_cache_maxsize=1)
    with dask.config.set(scheduler='single-threaded'):
        main(args)