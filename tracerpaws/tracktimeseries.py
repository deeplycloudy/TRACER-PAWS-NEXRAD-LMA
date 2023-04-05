"""
Support functions for working with the timeseries data file format created by TRACER-PAWS-NEXRAD-LMA
"""

import xarray as xr
import numpy as np
import pandas as pd

from .coords import feature_distance_from, cartesian_to_geographic

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
    
    Typical use, starting with track_timeseries_dataset opened by open_track_timeseries_dataset():
    ```
    features_by_track = track_timeseries_dataset.drop_dims(('x','y','time')).groupby('feature_parent_track_id')
    summed_features = features_by_track.sum(dim='feature')
    track_membership, track_counts = track_polarimetry(summed_features, zdr_thresh=0.0, kdp_thresh=0.0, flash_thresh=0.0)
    ```
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