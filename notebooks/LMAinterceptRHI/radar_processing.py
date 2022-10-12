import pyart
import numpy as np
def dealias_range_elev_derivatives(radar):
    """
    This function returns a pyart radar file with dealised radial velocity fields and spatial derivative in elevation (dVRde) and in range (dVRdr) 
    from a radar file read by pyart.
    Uses np.radians, np.gradient, np.ma.getdata, pyart.core.radar.Radar.add_field, pyart.correct.dealias_region_based
 
    """    
    swp_start = np.ma.getdata(radar.sweep_start_ray_index['data'])
    swp_end = np.ma.getdata(radar.sweep_end_ray_index['data'])    
    rrange = np.ma.getdata(radar.range['data']) # - 1D array [range]
    elevations = radar.elevation['data']   # - 1D array of elevations in deg                      
    el_fix = np.radians(elevations)                           
    # -- Dealias the velocity field
    dealias = pyart.correct.dealias_region_based(radar, vel_field = 'corrected_velocity')
    v_dealias = np.ma.getdata(dealias['data'])
    # -- dVRdr
    dVRdr = np.gradient(v_dealias, rrange, axis=1) # axis 1 = range axis
    # -- dVRde
    # -- I have to do like this because of the different sweeps
    dVRde = np.zeros_like(v_dealias)

    for i in np.arange(len(swp_start)):
        dVRde[swp_start[i],:] = (v_dealias[swp_start[i] + 1,:] - v_dealias[swp_start[i],:]) / (rrange * (el_fix[swp_start[i] + 1] - el_fix[swp_start[i]]))
        dVRde[swp_end[i],:] = (v_dealias[swp_end[i],:] - v_dealias[swp_end[i] - 1,:]) / (rrange * (el_fix[swp_end[i]] - el_fix[swp_end[i] - 1]))

    s = np.arange(len(el_fix))
    ss = np.delete(s,[swp_start,swp_end])
    for i in ss:
        dVRde[i,:] = (v_dealias[i + 1,:] - v_dealias[i - 1,:]) / (rrange * (el_fix[i + 1] - el_fix[i - 1]))

    # -- Adding fields to radar file
    dVRde = {'data' : dVRde}
    radar.add_field('dVRde', dVRde, replace_existing = True)

    dVRdr = {'data' : dVRdr}
    radar.add_field('dVRdr', dVRdr, replace_existing = True)

    radar.add_field('vel_dealias', dealias, replace_existing = True)

    return radar

from interp_funcs import centers_to_edges_1d, coords_2d
from coordinateSystems import RadarCoordinateSystem, TangentPlaneCartesianSystem
def r_z_centers_edges_mesh(data, sweep_idx):
    """
    This function takes a RHI radar file (.nc) read by xarray (data) and the sweep file index and 
    returns centers and edges for range and height in tangent plane cartesian system.
    """
    # -- Centers
    rr_c, az_c, el_c = coords_2d(data = data, sweep_idx = sweep_idx, centers = True)
    # -- Edges
    rr_e, az_e, el_e = coords_2d(data = data, sweep_idx = sweep_idx, centers = False)
    # -- Coordinates Systems
    ctrlat, ctrlon, ctralt = data.latitude['data'][0], data.longitude['data'][0], data.altitude['data'][0]

    # -- Radar Coordinate System RCS - Range, azimuth, elevation
    rcs = RadarCoordinateSystem(ctrlat, ctrlon, ctralt)
    tps = TangentPlaneCartesianSystem(ctrlat, ctrlon, ctralt)

    # -- Centers
    X_c, Y_c, Z_c = rcs.toECEF(rr_c, az_c, el_c)
    x_c, y_c, z_c = tps.fromECEF(X_c, Y_c, Z_c)
    x_c.shape = y_c.shape = z_c.shape = rr_c.shape 
    # -- Calculate range along tangent plane from x and y-coordinates
    r_c = np.sqrt(x_c**2 + y_c**2)  

    # -- Edges
    X_e, Y_e, Z_e = rcs.toECEF(rr_e, az_e, el_e)
    x_e, y_e, z_e = tps.fromECEF(X_e, Y_e, Z_e)
    x_e.shape = y_e.shape = z_e.shape = rr_e.shape 
    # -- Calculate range along tangent plane from x and y-coordinates
    r_e = np.sqrt(x_e**2 + y_e**2)
    
    return np.dstack((r_c, z_c)), np.dstack((r_e, z_e))
