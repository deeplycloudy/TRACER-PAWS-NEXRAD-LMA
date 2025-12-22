import numpy as np
import pyproj as proj4

def centers_to_edges(x):
    xedge=np.zeros(x.shape[0]+1)
    xedge[1:-1] = (x[:-1] + x[1:])/2.0
    dx = np.mean(np.abs(xedge[2:-1] - xedge[1:-2]))
    xedge[0] = xedge[1] - dx
    xedge[-1] = xedge[-2] + dx
    return xedge

# === Earth, projection, and grid specification from SBU team ===
def get_sbu_proj():
    """
    Set up projection data and map domain corresponding to the realtime radar
    grids to be produced for the TRACER and ESCAPE projects by the Stony Brook
    University cell tracking algorithm.

    Returns sbu_lla, sbu_map, x_edge, y_edge
    sbu_lla_, sbu_map: proj4 coordinate system objects using the SBU spherical
        earth and a stereographic map projection centered in Houston.
    x_edge, y_edge: 1D arrays of x and y coordinates for the realtime tracking
        domainin the SBU stereographic map projection.

    For any arrays of latitude and longitude, the map projection coordinates
        can be found with:
    sbu_lla, sbu_map, x_edge, y_edge = get_sbu_proj()
    dsx, dsy = proj4.transform(sbu_lla, sbu_map, longitude, latitude)

    """
    sbu_earth = 6367.0e3
    hou_ctr_lat, hou_ctr_lon = 29.4719, -95.0792
    dx = dy = 500.0
    nx, ny = 500, 500
    x = dx*(np.arange(nx, dtype='float') - nx/2) + dx/2
    y = dy*(np.arange(ny, dtype='float') - ny/2) + dy/2
    # x, y = np.meshgrid(x,y)
    sbu_map = proj4.crs.CRS(proj='stere', R=sbu_earth,
                         lat_0=hou_ctr_lat, lon_0=hou_ctr_lon)
    sbu_lla = proj4.crs.CRS(proj='latlong', R=sbu_earth)
    #a=sbu_earth, b=sbu_earth)
    x_edge = centers_to_edges(x)
    y_edge = centers_to_edges(y)
    return sbu_lla, sbu_map, x_edge, y_edge
# ======
