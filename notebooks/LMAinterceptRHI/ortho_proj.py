import pandas as pd
import numpy as np
import pyart
from lmatools.io.LMA_h5_file import LMAh5File
from lmatools.coordinateSystems import RadarCoordinateSystem, GeographicSystem, TangentPlaneCartesianSystem
import os
from radarlma2local import rcs_to_tps, geo_to_tps
from scipy.spatial.transform import Rotation

def close_sources(R, Z, lma_file_orthogonal, delta):
    """
    This function locates the point within a delta distance to the x-axis (min y) in the RHI scan (distance from radar R and height above radar Z).
    """
    a = lma_file_orthogonal[np.where(abs(lma_file_orthogonal[:,1]) < delta)]
    print('y-dist =', a[:,1])
    for i in np.arange(a[:,0].size):
        if a[i, 0] < 0: # not in the scan
            print('x-dist =', a[i,0])
            print('y-dist =', a[i,1])

    # R - horizontal
    R_cls = [R.flatten()[np.where(abs(R.flatten() - a[i,0]) == min(abs(R.flatten() - a[i,0])))][0] for i in np.arange(len(a[:,0]))]
    # Z - verical
    Z_cls = [R.flatten()[np.where(abs(R.flatten() - a[i,2]) == min(abs(R.flatten() - a[i,2])))][0] for i in np.arange(len(a[:,2]))]

    return np.asarray(R_cls), np.asarray(Z_cls), np.asarray(a[:,1])

def closest_int_pt(R, Z, lma_file_orthogonal):
    """
    This function locates the point closest (min y) to the RHI scan (distance from radar R and height above radar Z).
    """
    a = lma_file_orthogonal[np.where(abs(lma_file_orthogonal[:,1]) == min(abs(lma_file_orthogonal[:,1])))][0]
    print('y-dist =', a[1])
    if a[0] < 0:
        print('x-dist =', a[0])
        print('y-dist =', a[1])

    # R - horizontal
    R_close = R.flatten()[np.where(abs(R.flatten() - a[0]) == min(abs(R.flatten() - a[0])))][0]
    # Z - verical
    Z_close = Z.flatten()[np.where(abs(Z.flatten() - a[2]) == min(abs(Z.flatten() - a[2])))][0]

    return R_close, Z_close

def closest_pt_radarloc(radar, point, first_swp = False):
    """
    Given a radar file read by pyart and a value for (range,azimuth,elevation),
    it returns the closest point in the radar file to the specified location in the order [range, azimuth, elevation]
    """
    if first_swp == False:
        r = radar.range['data']
        az = radar.azimuth['data'][:]
        elev = radar.elevation['data'][:]
    else:
        r = radar.range['data']
        az = radar.azimuth['data'][radar.sweep_start_ray_index['data'][0] : radar.sweep_end_ray_index['data'][0]]
        elev = radar.elevation['data'][radar.sweep_start_ray_index['data'][0] : radar.sweep_end_ray_index['data'][0]]
    #
    cls_r_idx = np.where(abs(r - point[0]) == min(abs(r - point[0])))[0][0]
    #
    cls_az_idx = np.where(abs(az - point[1]) == min(abs(az - point[1])))[0][0]
    #
    cls_elev_idx = np.where(abs(elev - point[2]) == min(abs(elev - point[2])))[0][0]
    #
    return  cls_r_idx, cls_az_idx, cls_elev_idx


def rot_mat_lma(radar_file, lma_points, direction):
    """
    Given a radar file read by pyart and lma point in the (N,3) format in the TPCS x: pointing east, y: pointing north and z: local height,
    it returns the lma points rotated (N,3):x pointing along the RHI scan, y: orthogonal conterclockwise and z: local height.
    direction : * clockwise = 1
                * counter clockwise = -1
    """
    az = radar_file.azimuth['data'][0]
    ang_cart = np.deg2rad(450 - az)
    if ang_cart > np.pi/2:
        ang_cart = direction*ang_cart

    r = Rotation.from_rotvec([0, 0, ang_cart]) # - Counter clock wise direction

    rot_pts = r.apply(lma_points) # - (N,3)

    return rot_pts

def ortho_proj_lma(radar_file, lma_file):
    """
    Given a lma file read by lmatools and radar file read by pyart, it transforms both datasets to tangent plane coordinate system,
    and returns the lma sources coordinates rotated with x:pointing along the RHI scan, y:orthogonal to x, counterclockwise and z:local height.
    """
    Xlma,Ylma,Zlma = geo_to_tps(lma_file, radar_file)
    X, Y, Z = rcs_to_tps(radar_file)

    lon_ini1 = X[0,0]
    lat_ini1 = Y[0,0]
    lon_fin1 = X[0,-1]
    lat_fin1 = Y[0,-1]


    dlon1 = lon_fin1 - lon_ini1 # dx
    dlat1 = lat_fin1 - lat_ini1 # dy
    ds1 = np.array((dlon1,dlat1))
    norm_ds1 = np.linalg.norm(ds1)
    cross_ds1 = np.tensordot(ds1, ds1, (0,0))

    # LMA
    lma_file_n = np.column_stack((Xlma, Ylma))

    lma_file_loc_par = np.zeros(shape=(len(lma_file_n), 2))
    lma_file_loc_perp = np.zeros(shape=(len(lma_file_n), 2))
    lma_file_loc  = np.zeros(shape=(len(lma_file_n), 3))

    #
    # ##################################
    #
    #   (Xlma[i],Ylma[i]).ds1   .ds1
    #   ----------------------
    #        ds1 . ds1
    #
    # ##################################
    #
    lma_file_loc_tensor_x = np.tensordot(ds1,lma_file_n,(0,1))
    lma_file_loc_par = np.tensordot((lma_file_loc_tensor_x / cross_ds1 ),ds1,0)

    #
    # #######################################################################
    #
    #     (Xlma[i],Ylma[i])     _     (Xlma[i],Ylma[i]).ds1   .ds1
    #                                ----------------------
    #                                       ds1 . ds1
    #
    ##########################################################################
    #
    lma_file_loc_perp = lma_file_n - lma_file_loc_par

    #
    lma_file_loc[:,0] = np.sqrt(lma_file_loc_par[:,0]**2 + lma_file_loc_par[:,1]**2)
    lma_file_loc[:,1] = np.sqrt(lma_file_loc_perp[:,0]**2 + lma_file_loc_perp[:,1]**2)
    lma_file_loc[:,2] = Zlma


    return lma_file_loc

