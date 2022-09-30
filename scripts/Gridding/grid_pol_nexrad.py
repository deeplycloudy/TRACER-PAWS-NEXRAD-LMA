import argparse

parse_desc = """Grid single-site NEXRAD radar data after performing polarimetric
processing.

Data are automatically obtained from the AWS S3 NEXRAD archive based on the
site, start, and end times provided. After polarimetric processing, data in
CF-radial NetCDF format are saved to disk.

Each of these files is then gridded and saved to disk as a separate NetCDF file
in the same output directory.

Example
=======
python grid_pol_nexrad.py --site=KHGX --start=20170713_070000 \
  --end=20170713_080000 -o ./test

The command creates pairs of files like
KHGX20170713_070208_V06_grid.nc
KHGX20170713_070208_V06_proc.nc
in the test directory
"""


def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(dest='filenames',metavar='filename', nargs='*')
    parser.add_argument('-o', '--output_path',
                        metavar='filename template including path',
                        required=False, dest='outdir', action='store',
                        default='.', help='path in which to save data')
    parser.add_argument('--site', metavar='site', required=True,
                        dest='site', action='store',
                        help='NEXRAD site code, e.g., khgx')
    parser.add_argument('--start', metavar='yyyymmdd_hhmmss', required=True,
                        dest='start', action='store',
                        help='UTC start time, e.g., 20170704_080000')
    parser.add_argument('--end', metavar='yyyymmdd_hhmmss', required=True,
                        dest='end', action='store',
                        help='UTC start time, e.g., 20170704_090000')
    return parser

# End parsing #

import os
from copy import deepcopy

from data_utils import get_nexrad_keys, read_nexrad_key
import glob
import pyart
from dualpol import DualPolRetrieval # https://github.com/nasa/DualPol

def get_grid(radar):
    """ Returns grid object from radar object. """
    grid = pyart.map.grid_from_radars(
        radar, grid_shape=(31, 1001, 1001),
        grid_limits=((0, 15000), (-250000,250000), (-250000, 250000)),
        fields=['reflectivity','cross_correlation_ratio', 'differential_reflectivity', 'KDP_CSU', 'D0', 'NW', 'MU', 'MW', 'MI'],
        gridding_algo='map_gates_to_grid',
        h_factor=0., nb=0.6, bsp=1., min_radius=200.)
    return grid

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # Read S3 NEXRAD data, process polarimetry, and save to disk
    # The loop could be easily parallelized with the multiprocessing module.
    keys = get_nexrad_keys(args.site, start=args.start, end=args.end)
    radial_filenames = []
    for key in keys:
        fname = os.path.split(str(key))[1][:-1]
        if os.path.isfile(fname) == True :
            pass
        if os.path.isfile(fname) == False :
            print(key)
            radar = read_nexrad_key(key)
            zc = deepcopy(radar.fields['reflectivity']['data'])
            radar.add_field_like('reflectivity', 'ZC', zc,
                                 replace_existing=True)
            retrieve = DualPolRetrieval(radar, gs=250.0, dz='ZC',
                                      dr='differential_reflectivity',
                                      dp='differential_phase',
                                      rh='cross_correlation_ratio',
                                      fhc_T_factor=2, ice_flag=True,
                                      kdp_window=5, thresh_sdp=20.0, speckle=3,
                                      rain_method='hidro',
                                      qc_flag=True, verbose=True)
            radial_filename = os.path.join(args.outdir, fname + '_proc.nc')
            pyart.io.write_cfradial(radial_filename, radar)
            del radar
            radial_filenames.append(radial_filename)
    #radial_filenames = sorted(glob.glob('KHGX2022062*_proc.nc'))
    # create a grid.
    # No reason this needs to be in this script; it could be pulled out to a
    # separate script, with caution taken to decide on another strategy for 
    #now, it assuems that the filename contains _proc,
    # which was manually added above.
    # The loop could be easily parallelized with the multiprocessing module.
    grid_filenames = []
    for num, key in enumerate(radial_filenames):
        print('saving grid', num)
        print(key)
        radar = pyart.io.read_cfradial(key)
        grid = get_grid(radar)
        name = key.replace('_proc', '_grid')
        # name = os.path.join('Tgrid_' + str(num).zfill(3) + '.nc')
        grid_filenames.append(name)
        pyart.io.write_grid(name, grid)
        del radar, grid
