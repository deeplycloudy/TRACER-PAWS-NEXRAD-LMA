import argparse

parse_desc = """ Given an LMA data file with flash-sorted data, regrid it to the coordinate system
used by the SBU MAAS data synthesis and cell-tracking system, in both 2D and 3D as a function of
time.

Drops any old gridded data in the dataset.
"""

from lmalibtracer.coords.sbu import get_sbu_proj




def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(dest='filename',metavar='filename', nargs=1)
    parser.add_argument('-o', '--output_path',
                        metavar='filename template including path',
                        required=False, dest='outdir', action='store',
                        default='.', help='path in which to save data')
    parser.add_argument('-t', '--timestep', metavar='timestep', required=False,
                        dest='timestep', action='store', type=float, default=60,
                        help='gridding time interval, seconds')
    parser.add_argument('-n', '--minevents', metavar='minevents', required=False,
                        dest='minevents', action='store', type=int, default=5,
                        help='minimum events per flash')
    parser.add_argument('-s', '--subdivide', metavar='interval', required=False,
                        dest='subdivide_sec', action='store', type=float, default=3600,
                        help='output files at this subdivision interval (seconds)')

    return parser

import sys, glob
from copy import deepcopy
from pprint import pprint
from pathlib import Path
# Only in Py 3.10 or higher
# from itertools import pairwise
from itertools import tee

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


from lmalibtracer.coords.sbu import get_sbu_proj as get_coord_proj

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def parse_lma_filename(filename):
    parts = filename.split('_')
    duration_part = int(parts[3])
    start = datetime.datetime.strptime(parts[1]+parts[2], '%y%m%d%H%M%S')
    end = start + datetime.timedelta(seconds=duration_part)
    return start, end, duration_part

def process_one_chunk(ds, time_range, alt_range):
    # Project lon, lat to SBU map projection
    sbu_lla, sbu_map, x_edge, y_edge = get_coord_proj()
    sbu_dx = x_edge[1] - x_edge[0]
    sbu_dy = y_edge[1] - y_edge[0]
    lma_sbu_xratio = 1.0
    lma_sbu_yratio = 1.0
    trnsf_to_map = proj4.Transformer.from_crs(sbu_lla, sbu_map)
    trnsf_from_map = proj4.Transformer.from_crs(sbu_map, sbu_lla)

    if ds.dims['number_of_events'] > 0:
        have_events=True
        lmax, lmay = trnsf_to_map.transform(#sbu_lla, sbu_map,
                                     ds.event_longitude.data,
                                     ds.event_latitude.data)
        lma_initx, lma_inity = trnsf_to_map.transform(#sbu_lla, sbu_map,
                                     ds.flash_init_longitude.data,
                                     ds.flash_init_latitude.data)
        lma_ctrx, lma_ctry = trnsf_to_map.transform(#sbu_lla, sbu_map,
                                     ds.flash_center_longitude.data,
                                     ds.flash_center_latitude.data)
    else:
        have_events=False
        lmax, lmay = [],[]
        lma_initx, lma_inity = [],[]
        lma_ctrx, lma_ctry = [],[]
    ds['event_x'] = xr.DataArray(lmax, dims='number_of_events')
    ds['event_y'] = xr.DataArray(lmay, dims='number_of_events')
    ds['flash_init_x'] = xr.DataArray(lma_initx, dims='number_of_flashes')
    ds['flash_init_y'] = xr.DataArray(lma_inity, dims='number_of_flashes')
    ds['flash_ctr_x'] = xr.DataArray(lma_ctrx, dims='number_of_flashes')
    ds['flash_ctr_y'] = xr.DataArray(lma_ctry, dims='number_of_flashes')


    grid_config_3d = dict(
        grid_edge_ranges ={
            'grid_x_edge':(x_edge[0],x_edge[-1]+.001,sbu_dx*lma_sbu_xratio),
            'grid_y_edge':(y_edge[0],y_edge[-1]+.001,sbu_dy*lma_sbu_yratio),
            'grid_altitude_edge':alt_range,
            'grid_time_edge':time_range,
        },
        grid_center_names ={
            'grid_x_edge':'grid_x',
            'grid_y_edge':'grid_y',
            'grid_altitude_edge':'grid_altitude',
            'grid_time_edge':'grid_time',
        },
        event_coord_names = {
            'event_x':'grid_x_edge',
            'event_y':'grid_y_edge',
            'event_altitude':'grid_altitude_edge',
            'event_time':'grid_time_edge',
        },
        flash_init_names = {
            'flash_init_x':'grid_x_edge',
            'flash_init_y':'grid_y_edge',
            'flash_init_altitude':'grid_altitude_edge',
            'flash_time_start':'grid_time_edge',
        },
        flash_ctr_names = {
            'flash_ctr_x':'grid_x_edge',
            'flash_ctr_y':'grid_y_edge',
            'flash_center_altitude':'grid_altitude_edge',
            'flash_time_start':'grid_time_edge',
        }
    )
    # pprint(grid_config_3d)

    grid_config_2d = deepcopy(grid_config_3d)
    grid_config_2d['grid_edge_ranges'].pop('grid_altitude_edge')
    grid_config_2d['grid_center_names'].pop('grid_altitude_edge')
    grid_config_2d['event_coord_names'].pop('event_altitude')
    grid_config_2d['flash_init_names'].pop('flash_init_altitude')
    grid_config_2d['flash_ctr_names'].pop('flash_center_altitude')
    # pprint(grid_config_2d)


    grid_config_tz = deepcopy(grid_config_3d)
    grid_config_tz['grid_edge_ranges'].pop('grid_x_edge')
    grid_config_tz['grid_edge_ranges'].pop('grid_y_edge')
    grid_config_tz['grid_center_names'].pop('grid_x_edge')
    grid_config_tz['grid_center_names'].pop('grid_y_edge')
    grid_config_tz['event_coord_names'].pop('event_x')
    grid_config_tz['event_coord_names'].pop('event_y')
    grid_config_tz['flash_init_names'].pop('flash_init_x')
    grid_config_tz['flash_init_names'].pop('flash_init_y')
    grid_config_tz['flash_ctr_names'].pop('flash_ctr_x')
    grid_config_tz['flash_ctr_names'].pop('flash_ctr_y')
    # pprint(grid_config_tz)

    print("Creating regular grid")
    grid_ds = create_regular_grid(grid_config_3d['grid_edge_ranges'],
                                  grid_config_3d['grid_center_names'])
    ctrx, ctry = np.meshgrid(grid_ds.grid_x, grid_ds.grid_y)
    hlon, hlat = trnsf_from_map.transform(ctrx, ctry)

    grid_ds['lon'] = xr.DataArray(hlon, dims=['grid_y', 'grid_x'],
                    attrs={'standard_name':'longitude'})
    grid_ds['lat'] = xr.DataArray(hlat, dims=['grid_y', 'grid_x'],
                    attrs={'standard_name':'latitude'})


    grid_postfixes = ('tyx','tz','tzyx')
    all_grid_spatial_coords=(
        ['grid_time', None, 'grid_y', 'grid_x'],
        ['grid_time', 'grid_altitude', None, None],
        ['grid_time', 'grid_altitude', 'grid_y', 'grid_x']
        )
    all_grid_configs = (grid_config_2d, grid_config_tz, grid_config_3d)
    output_grids = {}
    for grid_postfix, grid_spatial_coords, grid_config in zip(
                grid_postfixes,
                all_grid_spatial_coords,
                all_grid_configs):

        print("Finding grid position for flashes for grid", grid_postfix)
        pixel_id_var = 'event_pixel_id'+'_'+grid_postfix
        ds_ev = assign_regular_bins(grid_ds, ds, grid_config['event_coord_names'],
            pixel_id_var=pixel_id_var, append_indices=True)

        event_spatial_vars = ('event_altitude', 'event_y', 'event_x')

        grid_ds = events_to_grid(
            ds_ev, grid_ds, min_points_per_flash=min_events_per_flash,
            pixel_id_var=pixel_id_var,
            event_spatial_vars=event_spatial_vars,
            grid_spatial_coords=grid_spatial_coords)
        both_ds = xr.combine_by_coords((grid_ds, ds))
        output_grids[grid_postfix] = both_ds

    return output_grids


def compress_all(nc_grids, min_dims=2):
    for var in nc_grids:
        if len(nc_grids[var].dims) >= min_dims:
            # print("Compressing ", var)
            nc_grids[var].encoding["zlib"] = True
            nc_grids[var].encoding["complevel"] = 4
            nc_grids[var].encoding["contiguous"] = False
    return nc_grids

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # '/data/Houston/LIGHTNING/6sensor_minimum/LYLOUT_220617_000000_86400_map500m.nc'
    filename = args.filename[0]
    path = Path(filename)
    base_filename = path.parts[-1]
    base_filename_no_extension = path.parts[-1].replace(path.suffixes[-1], '')
    output_path = Path(args.outdir)

    lma_start, lma_end, lma_duration = parse_lma_filename(base_filename)

    subdivide_sec = args.subdivide_sec
    grid_time_delta_sec = args.timestep
    min_events_per_flash = args.minevents

    alt_range = (0, 18e3+.001, 1.0e3)


    grid_dt = np.asarray(grid_time_delta_sec, dtype='m8[s]')
    # grid_t0 = np.datetime64(lma_start)
    # grid_t1 = np.datetime64(lma_end)
    print(lma_start, lma_end)



    # drop any old grid dimenions.
    ds = xr.open_dataset(filename)
    existing_grid_dims = [k for k in list(ds.dims.keys()) if 'grid' in k]

    # Explicitly load everything in our reduced dataset into memory, which
    # has a bit of a speed advantage.

    # Loading also works around a bug where deferring a load until after an empty
    # slice along the event dimension also defers decoding of datetimes. It will
    # eventually hit pd.to_timedelta(flat_num_dates.min(), delta) + ref_date;
    # in xarray/coding/times.py. That line does not work if there is no data,
    # since min(empty) is a ValueError. By loading up front, we force decode while
    # there is data and don't hit this problem. Before doing this, the whole program
    # ran until it tried to *write* the final NetCDF, which triggered the chain of
    # decoding for the final computation before re-encode and write.

    # import xarray as xr
    # import numpy as np
    # print(xr.__version__, np.__version__)
    # # Runs fine when starting from in-memory.
    # ds_empty = xr.Dataset({'time':xr.DataArray(np.asarray([], dtype='<M8[ns]'))})
    # ds_empty.to_netcdf('empty_time.nc')
    #
    # # Create a sample data file. Runs fine.
    # ds_times = xr.Dataset({'time':xr.DataArray(np.asarray(['2022-11-01', '2022-11-02', '2022-11-03'], dtype='<M8[ns]'))})
    # ds_times.to_netcdf('timedata.nc')
    # ds_times.close()
    #
    # # Read the sample data file, get an empty subset, and write. It crashes.
    # ds=xr.open_dataset('timedata.nc')
    # ds_reduced = ds[{'dim_0':slice(0,0)}]
    # ds_reduced.to_netcdf('time_nodata.nc')

    ds = ds.drop_dims(existing_grid_dims).load()


    bin_increment = np.timedelta64(subdivide_sec, 's')
    hourly_bins = np.arange(lma_start, lma_end, bin_increment)
    # print(hourly_bins)
    if np.datetime64(lma_end) > hourly_bins[-1]:
        hourly_bins = np.hstack([hourly_bins,[hourly_bins[-1]+bin_increment]])
    hourly_bin_labels = list(pairwise(hourly_bins))
    # for bl in hourly_bin_labels: print(bl)

    hourly_flash_groups = ds.groupby_bins(
        ds.flash_time_start, hourly_bins, labels=hourly_bin_labels)

    # Some times we have no data; caculate the empty set up once here.
    empty_slice = slice(0,0)
    empty_subset = ds[{'number_of_flashes':empty_slice,
                       'number_of_events':empty_slice}]

    # Index manually to work around the fact that intervals without data are
    # dropped when looping over the groupby (len(hourly_flash_groups) <= len(hourly_bin_labels))
    for hourly_index in hourly_bin_labels[:4]:
        time_range = (hourly_index[0], hourly_index[1], grid_dt)
        print("Grid time specs: ", time_range)
        try:
            flash_subset = hourly_flash_groups[hourly_index]#.load()
            self_consistent_subset = filter_flashes(flash_subset, prune=True)
            print(self_consistent_subset.event_time.min().data, self_consistent_subset.event_time.max().data)
        except KeyError:
            # no data at this time
            self_consistent_subset = empty_subset
            print('No data')
        gridded_chunk = process_one_chunk(self_consistent_subset, time_range, alt_range)

        output_file_basename = pd.Timestamp(hourly_index[0]).strftime(
            f'LYLOUT_%y%m%d_%H%M%S_{subdivide_sec:05d}_maas')
        print(output_file_basename)
        for grid_postfix, both_ds in gridded_chunk.items():
            out_filename = output_path.joinpath(output_file_basename+'_'+grid_postfix+'.nc')
            compress_all(both_ds).to_netcdf(out_filename)
        print('-----')
