#!/usr/bin/env python
# coding: utf-8

# # From VHF sources to lightning flashes
# 
# Using space-time criteria, we will group LMA data into flashes. We will also create a 2D gridded version of these flash data to look at VHF source density, flash extent density, and average flash area. 
# 
# Background reading: [Bruning and MacGorman (2013, JAS)](http://dx.doi.org/10.1175/JAS-D-12-0289.1) show how flash area is defined from the convex hull of the VHF point sources (Fig. 2) and show what flash extent density and average flash area look like for a supercell (Fig. 4). We'll use the same definitions here.

# In[1]:


import glob
import numpy as np
import datetime
import xarray as xr
import pandas as pd
import pyproj as proj4

from pyxlma.lmalib.io import read as lma_read
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats, filter_flashes
from pyxlma.lmalib.grid import  create_regular_grid, assign_regular_bins, events_to_grid
from pyxlma.plot.xlma_plot_feature import color_by_time, plot_points, setup_hist, plot_3d_grid, subset
from pyxlma.plot.xlma_base_plot import subplot_labels, inset_view, BlankPlot

import sys, glob

#from lmalibtracer.coords.sbu import get_sbu_proj as get_coord_proj


# ## Configurable Parameters 
# 
# These are the parameters that will be used by the flash sorting and gridding algorithms - it's a fairly small list.
# 
# These parameters are the ones we would expect to adjust when controlling for noise. We also require that a flash have more than one point!
# 
# 

# In[26]:


# filenames = sorted(glob.glob("/Users/kelcy/DATA/LYLOUT_22033*.dat.gz"))

filenames = sorted(glob.glob("/Users/kelcy/DATA/LMA/HLMA/220602/LYLOUT_220602_06*_0600.dat.gz"))
print(filenames)
# In[27]:


# Adjust this to match the length of the dataset we read in. It is used to set up 
# the total duration of the gridding along the time dimension.
duration_min = 60

# Source to flash 
chi2max = 1.0
stationsmin = 6
min_events_per_flash = 10

# There is a parameter to change the gridding from the 1 min default time resolution.
grid_time_delta_sec = 60 #60
resolution_m = 500 #1000
latlon_grid= True #True #False # False uses the stereographic coordinate grid.



lma_data, starttime = lma_read.dataset(filenames)
good_events = (lma_data.event_stations >= stationsmin) & (lma_data.event_chi2 <= chi2max)
lma_data = lma_data[{'number_of_events':good_events}]

dttuple = [starttime, starttime+datetime.timedelta(minutes=duration_min)]
# dttuple = lma_data.Datetime.min(), lma_data.Datetime.max()
tstring = 'LMA {}-{}'.format(dttuple[0].strftime('%H%M'),
                                  dttuple[1].strftime('%H%M UTC %d %B %Y '))
print(tstring)

print("Clustering flashes")
ds = cluster_flashes(lma_data)
print("Calculating flash stats")
ds = flash_stats(ds)
ds = filter_flashes(ds, flash_event_count=(min_events_per_flash, None))
# ds0 = ds.copy()
print(ds)




### KELCY TRYING TO GRID ON THE SAME GRID AS RADAR
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
def get_radar_proj():
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
#     proj_params = {'latitude_of_projection_origin': 34.93055725097656,
#          'longitude_of_projection_origin': -86.08361053466797,
#          '_CoordinateTransformType': 'Projection',
#          '_CoordinateAxes': 'x y z time',
#          '_CoordinateAxesTypes': 'GeoX GeoY Height Time',
#          'grid_mapping_name': 'azimuthal_equidistant',
#          'semi_major_axis': 6370997.0,
#          'inverse_flattening': 298.25,
#          'longitude_of_prime_meridian': 0.0,
#          'false_easting': 0.0,
#          'false_northing': 0.0}
    
    
    sbu_earth = 6367.0e3
    ctr_lat, ctr_lon = 29.47190094, -95.07873535
    
    dx = dy = 500.0
    nx, ny = 1001, 1001
    x = dx*(np.arange(nx, dtype='float') - nx/2) + dx/2
    y = dy*(np.arange(ny, dtype='float') - ny/2) + dy/2
    # x, y = np.meshgrid(x,y)
    radar_map = proj4.crs.CRS(proj='aeqd', R=sbu_earth,
                         lat_0=ctr_lat, lon_0=ctr_lon)
    radar_lla = proj4.crs.CRS(proj='latlong', R=sbu_earth)
    #a=sbu_earth, b=sbu_earth)
    x_edge = centers_to_edges(x)
    y_edge = centers_to_edges(y)
    return radar_lla, radar_map, x_edge, y_edge

print("Setting up grid spec")
grid_dt = np.asarray(grid_time_delta_sec, dtype='m8[s]')
grid_t0 = np.asarray(dttuple[0]).astype('datetime64[ns]')
grid_t1 = np.asarray(dttuple[1]).astype('datetime64[ns]')
time_range = (grid_t0, grid_t1+grid_dt, grid_dt)


radar_lla, radar_map, x_edge, y_edge = get_radar_proj()
    # Project lon, lat to SBU map projection

radar_dx = x_edge[1] - x_edge[0]
radar_dy = y_edge[1] - y_edge[0]
lma_radar_xratio = resolution_m/radar_dx
lma_radar_yratio = resolution_m/radar_dy
trnsf_to_map = proj4.Transformer.from_crs(radar_lla, radar_map)
trnsf_from_map = proj4.Transformer.from_crs(radar_map, radar_lla)
lmax, lmay = trnsf_to_map.transform(#sbu_lla, sbu_map,
                                 ds.event_longitude.data,
                                 ds.event_latitude.data)
lma_initx, lma_inity = trnsf_to_map.transform(#sbu_lla, sbu_map,
                                 ds.flash_init_longitude.data,
                                 ds.flash_init_latitude.data)
lma_ctrx, lma_ctry = trnsf_to_map.transform(#sbu_lla, sbu_map,
                                 ds.flash_center_longitude.data,
                                 ds.flash_center_latitude.data)
ds['event_x'] = xr.DataArray(lmax, dims='number_of_events')
ds['event_y'] = xr.DataArray(lmay, dims='number_of_events')
ds['flash_init_x'] = xr.DataArray(lma_initx, dims='number_of_flashes')
ds['flash_init_y'] = xr.DataArray(lma_inity, dims='number_of_flashes')
ds['flash_ctr_x'] = xr.DataArray(lma_ctrx, dims='number_of_flashes')
ds['flash_ctr_y'] = xr.DataArray(lma_ctry, dims='number_of_flashes')

grid_edge_ranges ={
        'grid_x_edge':(x_edge[0],x_edge[-1]+.001,radar_dx*lma_radar_xratio),
        'grid_y_edge':(y_edge[0],y_edge[-1]+.001,radar_dy*lma_radar_yratio),
    #     'grid_altitude_edge':alt_range,
        'grid_time_edge':time_range,
    }
grid_center_names ={
        'grid_x_edge':'grid_x',
        'grid_y_edge':'grid_y',
    #     'grid_altitude_edge':'grid_altitude',
        'grid_time_edge':'grid_time',
    }

event_coord_names = {
        'event_x':'grid_x_edge',
        'event_y':'grid_y_edge',
    #     'event_altitude':'grid_altitude_edge',
        'event_time':'grid_time_edge',
    }

flash_ctr_names = {
        'flash_init_x':'grid_x_edge',
        'flash_init_y':'grid_y_edge',
    #     'flash_init_altitude':'grid_altitude_edge',
        'flash_time_start':'grid_time_edge',
    }
flash_init_names = {
        'flash_ctr_x':'grid_x_edge',
        'flash_ctr_y':'grid_y_edge',
    #     'flash_center_altitude':'grid_altitude_edge',
        'flash_time_start':'grid_time_edge',
    }


print("Creating regular grid")
grid_ds = create_regular_grid(grid_edge_ranges, grid_center_names)

ctrx, ctry = np.meshgrid(grid_ds.grid_x, grid_ds.grid_y)
hlon, hlat = trnsf_from_map.transform(ctrx, ctry)
    # Add lon lat to the dataset, too.
ds['lon'] = xr.DataArray(hlon, dims=['grid_y', 'grid_x'],
                    attrs={'standard_name':'longitude'})
ds['lat'] = xr.DataArray(hlat, dims=['grid_y', 'grid_x'],
                    attrs={'standard_name':'latitude'})

print("Finding grid position for flashes")
pixel_id_var = 'event_pixel_id'
ds_ev = assign_regular_bins(grid_ds, ds, event_coord_names,
    pixel_id_var=pixel_id_var, append_indices=True)

grid_spatial_coords=['grid_time', None, 'grid_y', 'grid_x']
event_spatial_vars = ('event_altitude', 'event_y', 'event_x')
grid_ds = events_to_grid(ds_ev, grid_ds, min_points_per_flash=3,
                         pixel_id_var=pixel_id_var,
                         event_spatial_vars=event_spatial_vars,
                         grid_spatial_coords=grid_spatial_coords)


# In[ ]:


grid_ds


# In[ ]:


# Let's combine the flash and event data with the gridded data into one giant data structure.
both_ds = xr.combine_by_coords((grid_ds, ds))
print(both_ds)




# ## Finally, write the data.
# 
# Once we save the data to disk, we can reload the data an re-run any of the plots above without reprocessing everything. We'll make files like this from the post-processed LMA data for each day during ESCAPE/TRACER, and they will be one of our data deliverables to the NCAR EOL catalog, in accordance with the ESCAPE data management plan as proposed in the grant.

# In[ ]:


if True:
    print("Writing data")
    duration_sec = (dttuple[1]-dttuple[0]).total_seconds()
    if latlon_grid:
        date_fmt = "LYLOUT_%y%m%d_%H%M%S_{0:04d}_grid.nc".format(int(duration_sec))
    else:
        date_fmt = "LYLOUT_%y%m%d_%H%M%S_{0:04d}_map{1:d}m.nc".format(
                        int(duration_sec), resolution_m)
    outfile = dttuple[0].strftime(date_fmt)

    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in both_ds.data_vars}
    both_ds.to_netcdf(outfile, encoding=encoding)


