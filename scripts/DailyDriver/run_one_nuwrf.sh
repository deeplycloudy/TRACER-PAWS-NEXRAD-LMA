#!/bin/bash

# In the shell, or in the script calling this script, set these environment variables.
# FULLDATE="2022/06/02", and the others are float km in strings, e.g., $MELTLEVEL="4.4"
echo $FULLDATE#, $START, $END #, $MELTLEVEL, $HMGFRZLVL
echo $SPD
# Convert to use the unix date command and format strings so we only have to enter once
# export DATE2=`date -d $FULLDATE +%Y%m%d%H`
export DATE=`date -d $FULLDATE +%Y%m%d`
export SPEED=$SPD

# python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseriesPOLARRIS2.py \
#     --polarrispath=/home/jovyan/efs/tracer/POLARRIS_GRIDDED/$DATE/ \
#     --nuwrfpath=/home/jovyan/efs/tracer/NUWRF/$DATE/ \
#     --tobacpath=/home/jovyan/efs/tracer/TOBAC_POLARRIS/POLARRIS_tobac_Save_$DATE2/ --type="POLARRIS"
    
# python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseriesNUWRF.py \
#     --nuwrfpath=/home/jovyan/efs/tracer/NUWRF/$DATE/ \
#     --tobacpath=/home/jovyan/efs/tracer/TOBAC_NUWRF/forecast_runs/NUWRF_tobac_Save_$DATE2/ --type="NUWRF"

python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Tracking/knb_tobac_tracking.py \
  --path /home/jovyan/efs/tracer/NEXRAD/$DATE/ \
  --threshold=15 --speed=$SPEED --site=KHGX --type=NEXRAD


# python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Tracking/knb_tobac_tracking.py \
#   --path /home/jovyan/efs/tracer/POLARRIS_GRIDDED/$DATE/ \
#   --threshold=15 --speed=1.0 --site=KHGX --type=POLARRIS #--extra_vars="FRZLVL"

# python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Tracking/knb_tobac_plotting.py \
#   --path=/home/jovyan/efs/tracer/POLARRIS_GRIDDED/$DATE/ \ 
#   --tobacpath=/home/jovyan/efs/tracer/TOBAC_POLARRIS/POLARRIS_tobac_Save_$DATE2/ \
#   --type=POLARRIS --lat=29.4719 --lon=-95.0792 --dxy=0.5
# # for w in range(7,9):
#     for x in range(0,24):
#         command = f"python grid_polarris.py --path='/efs/tracer/NUWRF/2022070700/VCP12_CfRadial_2022_07{w:02d}_{x:02d}*.nc' "

#         os.system(command)



# for w in {$START,$MID,$END}
# do
#     for x in {00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
#     do 
#         echo "$w$x"
#         python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Gridding/grid_polarris.py --path=/efs/tracer/NUWRF/$DATE/VCP12_CfRadial_2022_060$w\_$x*.nc
#     done
# done

# kernprof -l /home/jovyan/code/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
# MALLOC_TRIM_THRESHOLD_=0 python /home/jovyan/code/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
# MALLOC_TRIM_THRESHOLD_=0 python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
# --path=/home/jovyan/efs/tracer/NEXRAD/$DATE/ \
# --tobacpath=/home/jovyan/efs/tracer/NEXRAD/tobac_Save_$DATE/ \
# --lmapath=/home/jovyan/efs/tracer/LIGHTNING/$LMAMONTH/6sensor_minimum/ \
# --meltinglevel=$MELTLEVEL --freezinglevel=$HMGFRZLVL --type="NEXRAD" 


# KHGX is 20 km from both CSAPR and the ARM site, and ANC is 65 km. So storms within 60 km of CSAPR are 
# within 80 km of KHGX. That only goes 15 km beyond the ANC site, so we choose a 90 km max range to get 
# 70 km from CSAPR and 25 km beyond the ANC site. That is 60% of 150 km.

# For a 15 km max storm top, tan theta = 15/90, that is a 9 degree elevation angle. 
# 88D VCP 215 has 10 elevation angles up to that altitude at that range, and has about 
# eight up to about 10 km altitude at that range, with minimum gaps in the beams. It's about 3-4
# elevtion cuts between 5-10 km, which is ok enough to get polarimetric columns.
# |
# | 15 km                   theta \
# |________________________________
#                90 km

# python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/tobac_track_histograms.py \
#   --trackpath=/efs/tracer/NEXRAD/tobac_Save_$DATE/Track_features_merges.nc \
#   --timeseriespath=/efs/tracer/NEXRAD/tobac_Save_$DATE/timeseries_data_melt$MELTM.nc \
#   --referencegridpath=$ONERADARGRID \
#   --output_path=/efs/tracer/NEXRAD/tobac_Save_$DATE/ \
#   --distance=90.0 

