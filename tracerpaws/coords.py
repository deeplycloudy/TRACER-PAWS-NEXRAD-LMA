import warnings

import numpy as np

try:
    import pyproj

    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

    

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