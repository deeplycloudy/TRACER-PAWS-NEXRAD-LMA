#Matthew Miller, 08-16-2022

#This script can be edited to loop through a day's worth of processing 1 hour at a time, in series. Seems to be more stable than doing chunks at a time, 
#and requires less babysitting.

import os
import sys


#Edit start of range to change start hour. Shouldn't need to edit end, as leaving it at 24 means it will process through 235959. 0 is the recommended start of range.

# for x in range(23,24):
# 	for y in range(0,7):
# 		command = f"python grid_pol_nexrad.py --site=KHGX --start=20220716_{x:02d}{y:1d}000 --end=20220716_{x:02d}{y:1d}959"
# 	#print(command)
	
# 		os.system(command)
        # python grid_polarris.py --path="/efs/tracer/NUWRF/2022060300/VCP12_CfRadial_2022_06*.nc'"
for w in range(30..31):
    for x in range(0,24):
        command = f"python grid_polarris.py --path='/efs/tracer/NUWRF/2022093000/VCP12_CfRadial_2022_09{w:02d}_{x:02d}*.nc' "

        os.system(command)


# example: python grid_pol_nexrad.py --site=KHGX --start=20220801_180000 --end=20220801_185959

