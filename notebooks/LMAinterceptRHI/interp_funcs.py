import numpy as np
def centers_to_edges_1d(x):
    """
    This function takes an array x with centers locations and returns an array with edges locations.
    """
    beam_sp = np.zeros_like(x)
    beam_sp = (x[1:] - x[:-1])/2 # -- It does not assume all the adjacent spacing is equal
    edges = np.zeros(len(x)+1)
    edges[0] = x[0] - beam_sp[0]
    edges[1:-1] = x[:-1] + beam_sp
    edges[-1] = x[-1] + beam_sp[-1]
    return edges

def coords_2d(data, sweep_idx, centers = True):
    """
    This function takes radar data read by xarray and returns range, azimuth and elevation centers (True) or edges (False) coordinates for a sweep index.
    """
    sel = [int(data.sweep_start_ray_index['data'][sweep_idx]), int(data.sweep_end_ray_index['data'][sweep_idx])]
    if centers == True:
        r_n = data.range['data']
        az_n = data.azimuth['data'][sel[0] : sel[1]+1]
        el_n = data.elevation['data'][sel[0] : sel[1]+1]
    else:
        r_n = centers_to_edges_1d(data.range['data'])  
        az_n = centers_to_edges_1d(data.azimuth['data'][sel[0] : sel[1]+1])  # -- Select values of sweep one before converting to edges
        el_n = centers_to_edges_1d(data.elevation['data'][sel[0] : sel[1]+1])
    
    r_mesh, el_mesh = np.meshgrid(r_n, el_n)
    r_mesh, az_mesh = np.meshgrid(r_n, az_n)
    return r_mesh, az_mesh, el_mesh

from scipy.spatial import cKDTree
def oban(points, values, xi, weight_func, search_radius):
    """
    This function interpolates values for xi locations based on values available at known points with a weighted function within a search radius. 
    points: N,2 data point locations
    values: N data values
    xi: M,2 analysis locations
    weight_func is a function that accepts a single argument r that is the distance between the analysis location and all points within 
    search_radius of the analysis location.
    """
    # Find all points in the vicinity of each analysis location in xi
    tree = cKDTree(points)
    query = tree.query_ball_point(xi, search_radius)
    analysis = np.zeros(xi.shape[0])
    # This is linear (times the typical neighborhood size) in the number of analysis points
    for i, (analysis_point, neighbors) in enumerate(zip(xi, query)):
        data = values[neighbors]
        data_locations = points[neighbors,:]
        # use data, data_locations, analysis_point, and weight_func to fill in the rest of the analysis
        rr = np.sqrt(np.sum((data_locations - analysis_point)**2, axis=1))
        W = weight_func(rr)
        mask_nan = np.isnan(data)  # mask out nan so sum works with nan values
        analysis[i] = np.sum(data[~mask_nan] * W[~mask_nan]) / np.sum(W[~mask_nan])
    return analysis

def barnes(r, k = None):
    """ 
    This function returns the weights as a funcion of distance r
    r has units of distance, and k is the dimensional weight parameter kappa
    kappa has units of distance squared.        
    """
    W = np.exp(-r**2 / k)
    return W


def weighted_mean(values, weight_idx, N, weights):
    """
    Returns weighted mean given the values, values related to the weights, the weights and the number of elements.
    """
    sum_f = 0
    w_f = 0
    for i in np.arange(N):
        sum_f = sum_f + weights[int(weight_idx[i])] * values[i]
        w_f = w_f + weights[int(weight_idx[i])]
    
    return sum_f/w_f

def weighted_std(values, weight_idx, weight_avg, N, weights):
    """
    Returns weighted standard deviation given the values, weighted average, values related to the weights, the weights and the number of elements.
    """
    sum_f = 0
    w_f = 0
    for i in np.arange(N):
        sum_f = sum_f + weights[int(weight_idx[i])] * ((values[i] - weight_avg)**2)
        w_f = w_f + weights[int(weight_idx[i])]

    return np.sqrt(sum_f / ((N-1/N) * w_f))

import scipy as sp
def interp_avg_std(df):
    """
    This function returns lists of the average, distance weighted average, standard deviation, and distance weighted standard deviation 
    for turbulence and distance from the 1st source from the input dataframe containing turbulence, 
    distance from interception and distance from the 1st source obtained from different interpolation methods
    for all the sources selected within the threshold.
    """
    
    # -- Weighted values for turbulence and distance from the first source
    weights = sp.signal.gaussian(200, 40)[100:]
    
    df_tur_weight_avg = weighted_mean(values = df.turbulence.values, 
                                      weight_idx = df.dist_itp.values, 
                                      N = len(df.turbulence.values),
                                      weights = weights)
    df_tur_weight_std = weighted_std(values = df.turbulence.values, 
                                     weight_idx = df.dist_itp.values, 
                                     weight_avg = df_tur_weight_avg,
                                     N = len(df.turbulence.values),
                                     weights = weights)

    df_dist1_weight_avg = weighted_mean(values = df.dist_1.values/1000,
                                        weight_idx = df.dist_itp.values, 
                                        N = len(df.turbulence.values),
                                        weights = weights)
    df_dist1_weight_std = weighted_std(values = df.dist_1.values/1000,
                                       weight_idx = df.dist_itp.values,
                                       weight_avg = df_dist1_weight_avg,
                                       N = len(df.turbulence.values),
                                       weights = weights)
    
    # -- Regular mean and standard deviation
    # - mean
    rm_df = np.mean(df.dist_1/1000.) 
    zm_df = np.mean(df.turbulence)
    # - std
    rstd_df = np.std(df.dist_1/1000.)
    zstd_df = np.std(df.turbulence)

    # - Mean, Weighted mean, standard deviation, weighted standard deviation for Nearest Neighbor
    # - x = dist 1 source, y = turbulence
    mean_df = list((rm_df, zm_df))
    wmean_df = list((df_dist1_weight_avg, df_tur_weight_avg))
    std_df = list((rstd_df, zstd_df))
    wstd_df = list((df_dist1_weight_std, df_tur_weight_std))

    return mean_df, std_df, wmean_df, wstd_df

