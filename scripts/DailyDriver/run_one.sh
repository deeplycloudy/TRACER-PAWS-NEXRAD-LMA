#!/bin/bash

# In the shell, or in the script calling this script, set these environment variables.
# FULLDATE="2022/06/02", and the others are float km in strings, e.g., $MELTLEVEL="4.4"
echo $FULLDATE, $MELTLEVEL, $HMGFRZLVL

# Convert to use the unix date command and format strings so we only have to enter once
export MONTHNAME=`date -d $FULLDATE +%B`
export LMAMONTH=`echo $MONTHNAME |  tr '[:upper:]' '[:lower:]'`
export LMADATE=`date -d $FULLDATE +%y%m%d`
export DATE=`date -d $FULLDATE +%Y%m%d`
# Convert melting level to meters
export MELTM=$(echo "$MELTLEVEL" | sed -r 's/\.//g' | sed 's/$/00/')

# Sample grid for projection information on that date (doesn't actually vary)
export ONERADARGRID=`ls -t /efs/tracer/NEXRAD/$DATE/KHGX2022*grid.nc | head -n1`

#python /home/jovyan/code/TRACER-PAWS-NEXRAD-LMA/scripts/Tracking/knb_tobac_tracking.py \
#  --path /home/jovyan/efs/tracer/NEXRAD/$DATE/ \
#  --threshold=15 --speed=1.0 --site=KHGX --type=NEXRAD


# kernprof -l /home/jovyan/code/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
# MALLOC_TRIM_THRESHOLD_=0 python /home/jovyan/code/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
MALLOC_TRIM_THRESHOLD_=0 echo python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
--path=/home/jovyan/efs/tracer/NEXRAD/$DATE/ \
--tobacpath=/home/jovyan/efs/tracer/NEXRAD/tobac_Save_$DATE/ \
--lmapath=/home/jovyan/efs/tracer/LIGHTNING/$LMAMONTH/6sensor_minimum/ \
--meltinglevel=$MELTLEVEL --freezinglevel=$HMGFRZLVL --type="NEXRAD" 


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

echo python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/tobac_track_histograms.py \
  --trackpath=/efs/tracer/NEXRAD/tobac_Save_$DATE/Track_features_merges.nc \
  --timeseriespath=/efs/tracer/NEXRAD/tobac_Save_$DATE/timeseries_data_melt$MELTM.nc \
  --referencegridpath=$ONERADARGRID \
  --output_path=/efs/tracer/NEXRAD/tobac_Save_$DATE/ \
  --distance=90.0 

