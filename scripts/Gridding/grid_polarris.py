#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import os
#!/usr/bin/env python
# coding: utf-8

import argparse
import shutil
from copy import deepcopy
import cartopy.crs as ccrs
import shapely
import pyart

import pandas as pd
import glob

# Disable a couple of warnings:
import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
warnings.filterwarnings("ignore", category=FutureWarning, append=True)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)



parse_desc = """Find features, track, and plot all nexrad data in a given destination folder

The path is a string destination
Files in path must have a postfix of 'grid.nc'. 
threshold is the tracking threshold in dbz
speed is the tracking speed in tobac units. 
Site is a string NEXRAD location


Example
=======
python grid_polarris.py --path="/Users/kelcy/DATA/20220807_toshi/POLARRIS/" 

"""

def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument('--path',metavar='path', required=True,dest='path',
                        action = 'store',help='path in which the data is located')

    return parser

# End parsing #




def get_grid(radar):
    """ Returns grid object from radar object. """
    grid = pyart.map.grid_from_radars(
        radar, grid_shape=(31, 1001, 1001),
        grid_limits=((0, 15000), (-250000,250000), (-250000, 250000)),
        fields=['CZ','DR','KD','RH','VR'],#,'W'],
        gridding_algo='map_gates_to_grid',
        h_factor=0., nb=0.6, bsp=1., min_radius=200.)
    return grid


# In[ ]:



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    

    arq = sorted(glob.glob(args.path))#+'VCP12*.nc'))
    # radar = pyart.io.read_cfradial(arq[0])
    new_vars = args.path.split('/')
    print(new_vars)
    new_vars[3] = 'POLARRIS_GRIDDED'
    new_vars = new_vars[:-1]
    #print(new_vars)
    new_path = "/".join(new_vars)
    print(new_path)

# In[ ]:


    filenames = []
    for num, key in enumerate(arq):
        # if num < 644:
        #     continue

        print(key)
        print('saving grid', num)
        radar = pyart.io.read_cfradial(key)
        grid = get_grid(radar)
        fname = os.path.split(str(key))[1][:-3]
        name = os.path.join(new_path +'/'+ fname + '_grid2.nc')
        pyart.io.write_grid(name, grid)
        del radar, grid

