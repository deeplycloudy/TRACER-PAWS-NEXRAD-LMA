#/bin/bash

#conda activate tracer_env

# Default melting and homogeneous freezing level.
export MELTLEVEL="4.4"
export HMGFRZLVL="10.6"

# Parameters that vary by day
# export FULLDATE="2022/06/02"
# export MELTLEVEL="4.7"

# # Parameters that vary by day
# export FULLDATE="2022/06/04"
# export MELTLEVEL="4.4"

# # Parameters that vary by day  - DOES NOT EXIST
# export FULLDATE="2022/06/16"
# export MELTLEVEL="4.8"

# # Parameters that vary by day
# export FULLDATE="2022/06/17"
# export MELTLEVEL="4.6"

# # Parameters that vary by day  - DOES NOT EXIST
# export FULLDATE="2022/06/21"
# export MELTLEVEL="5.1"

# # Parameters that vary by day
# export FULLDATE="2022/06/22"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/07/02"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/07/06"
# export MELTLEVEL="5.0"

# # Parameters that vary by day
# export FULLDATE="2022/07/12"
# export MELTLEVEL="5.2"

# # Parameters that vary by day
# export FULLDATE="2022/07/13"
# export MELTLEVEL="4.7"

# # Parameters that vary by day
# export FULLDATE="2022/07/14"
# export MELTLEVEL="4.7"

# # Parameters that vary by day
# export FULLDATE="2022/07/28"
# export MELTLEVEL="4.8"

# # Parameters that vary by day
# export FULLDATE="2022/07/29"
# export MELTLEVEL="4.8"

# # Parameters that vary by day
# export FULLDATE="2022/08/01"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/08/03"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/08/02"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/08/06"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/08/07"
# export MELTLEVEL="4.7"

# # Parameters that vary by day
# export FULLDATE="2022/08/08"
# export MELTLEVEL="4.7"

# # Parameters that vary by day
# export FULLDATE="2022/08/13"
# export MELTLEVEL="5.2"

# # Parameters that vary by day
# export FULLDATE="2022/08/21"
# export MELTLEVEL="5.0"

# # Parameters that vary by day
# export FULLDATE="2022/08/25"
# export MELTLEVEL="5.0"

# # Parameters that vary by day
# export FULLDATE="2022/08/27"
# export MELTLEVEL="5.1"

# # Parameters that vary by day
# export FULLDATE="2022/08/31"
# export MELTLEVEL="4.8"

# # Parameters that vary by day
export FULLDATE="2022/09/01"
# export MELTLEVEL="4.9"

# # Parameters that vary by day
# export FULLDATE="2022/09/15"
# export MELTLEVEL="4.7"

# # Parameters that vary by day
# export FULLDATE="2022/09/17"
# export MELTLEVEL="4.8"

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
MALLOC_TRIM_THRESHOLD_=0 python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/ecb_tobac_timseries.py \
--path=/home/jovyan/efs/tracer/NEXRAD/$DATE/ \
--tobacpath=/home/jovyan/efs/tracer/NEXRAD/tobac_Save_$DATE/ \
--lmapath=/home/jovyan/efs/tracer/LIGHTNING/$LMAMONTH/6sensor_minimum/ \
--meltinglevel=$MELTLEVEL --freezinglevel=$HMGFRZLVL --type="NEXRAD" 
#> $DATE.timeseries.log


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

python /home/jovyan/efs/Jupyter_NBs/TRACER-PAWS-NEXRAD-LMA/scripts/Analysis/tobac_track_histograms.py \
  --trackpath=/efs/tracer/NEXRAD/tobac_Save_$DATE/Track_features_merges.nc \
  --timeseriespath=/efs/tracer/NEXRAD/tobac_Save_$DATE/timeseries_data_melt$MELTM.nc \
  --referencegridpath=$ONERADARGRID \
  --output_path=/efs/tracer/NEXRAD/tobac_Save_$DATE/ \
  --distance=90.0 
  #> $DATE.histograms.log
