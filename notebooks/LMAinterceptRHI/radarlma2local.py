import numpy as np
import pyart
from lmatools.io.LMA_h5_file import LMAh5File
from coordinateSystems import RadarCoordinateSystem, GeographicSystem, TangentPlaneCartesianSystem
def rcs_to_tps(radar):
    """
    Given a rhi radar file read by pyart in range elevation azimuth coordinates,
    it returns the tangent plane coordinates X(pointing east),Y(pointing north),Z(local height) of it.    
    """

    # Coordinates Systems
    ctrlat, ctrlon, ctralt = np.ma.getdata(radar.latitude['data'][0]),np.ma.getdata(radar.longitude['data'][0]),np.ma.getdata(radar.altitude['data'][0]) 
    rcs = RadarCoordinateSystem(ctrlat, ctrlon, ctralt)
    tps = TangentPlaneCartesianSystem(ctrlat, ctrlon, ctralt)
    
    idx_start = radar.sweep_start_ray_index['data']
    idx_end = radar.sweep_end_ray_index['data']

    # - Elevations, azimuth, range

    l_elev = len(idx_start)
    l_az = len(radar.azimuth['data'])
    l_r = len(radar.range['data'])

    r = np.zeros((l_az, l_r))
    r[:,] = radar.range['data']

    els = radar.elevation['data']
    els = np.tensordot(els, np.ones_like(r[0,:]), axes=0)

    azi = radar.azimuth['data']
    az = np.tensordot(azi, np.ones_like(r[0,:]), axes=0)

    a, b, c = rcs.toECEF(r,az,els)
    abc = np.vstack((a,b,c))
    # ECEF to TPS
    n = tps.toLocal(abc)
    X = n[0,:]
    Y = n[1,:]
    Z = n[2,:]

    X = np.reshape(X,  (l_az, l_r))
    Y = np.reshape(Y,  (l_az, l_r))
    Z = np.reshape(Z,  (l_az, l_r))
    
    return X, Y, Z

def geo_to_tps(lma_file, radar_file):
    """
    Given a lma file read by lmatools in latitude, longitude, altitude coordinates,it returns the tangent plane coordinates Xlma(pointing east),Ylma(pointing north),Z(local height) of it.
    """
    # Coordinates Systems - radar
    ctrlat, ctrlon, ctralt = np.ma.getdata(radar_file.latitude['data'][0]),np.ma.getdata(radar_file.longitude['data'][0]), np.ma.getdata(radar_file.altitude['data'][0]) 

    # GeographicSystem GEO - Lat, lon, alt
    geo = GeographicSystem()
    # Tangent Plane Cartesian System TPS - 
    tps = TangentPlaneCartesianSystem(ctrlat, ctrlon, ctralt)

    # GEO to TPS

#     d, e, h = geo.toECEF(lma_file.lon.values,lma_file.lat.values,lma_file.alt.values)
    d, e, h = geo.toECEF(lma_file.event_longitude.data, lma_file.event_latitude.data, lma_file.event_altitude.data)
    deh = np.vstack((d,e,h))
    m = tps.toLocal(deh)
    Xlma = m[0,:]
    Ylma = m[1,:]
    Zlma = m[2,:]
    
    return Xlma,Ylma,Zlma

