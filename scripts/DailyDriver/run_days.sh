#!/bin/bash

# Default melting and homogeneous freezing level.
export MELTLEVEL="4.4"
export HMGFRZLVL="10.6"

# Each day is listed below. Uncomment at least the FULLDATE and ./run_one.sh
# lines to run that day.

export FULLDATE="2022/06/02"
export MELTLEVEL="4.7"
./run_one.sh &

export FULLDATE="2022/06/04"
export MELTLEVEL="4.4"
./run_one.sh &

# DOES NOT EXIST
# export FULLDATE="2022/06/16"
# export MELTLEVEL="4.8"
# ./run_one.sh &

export FULLDATE="2022/06/17"
export MELTLEVEL="4.6"
./run_one.sh &

# DOES NOT EXIST
# export FULLDATE="2022/06/21"
# export MELTLEVEL="5.1"
# ./run_one.sh &

export FULLDATE="2022/06/22"
export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/07/02"
export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/07/06"
export MELTLEVEL="5.0"
./run_one.sh &

export FULLDATE="2022/07/12"
export MELTLEVEL="5.2"
./run_one.sh &

export FULLDATE="2022/07/13"
export MELTLEVEL="4.7"
./run_one.sh &

export FULLDATE="2022/07/14"
export MELTLEVEL="4.7"
./run_one.sh &

export FULLDATE="2022/07/28"
export MELTLEVEL="4.8"
./run_one.sh &

export FULLDATE="2022/07/29"
export MELTLEVEL="4.8"
./run_one.sh &

export FULLDATE="2022/08/01"
export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/08/03"
export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/08/02"
export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/08/06"
export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/08/07"
export MELTLEVEL="4.7"
./run_one.sh &

export FULLDATE="2022/08/08"
# export MELTLEVEL="4.7"
./run_one.sh &

export FULLDATE="2022/08/13"
# export MELTLEVEL="5.2"
./run_one.sh &

export FULLDATE="2022/08/21"
# export MELTLEVEL="5.0"
./run_one.sh &

export FULLDATE="2022/08/25"
# export MELTLEVEL="5.0"
./run_one.sh &

export FULLDATE="2022/08/27"
# export MELTLEVEL="5.1"
./run_one.sh &

export FULLDATE="2022/08/31"
# export MELTLEVEL="4.8"
./run_one.sh &

export FULLDATE="2022/09/01"
# export MELTLEVEL="4.9"
./run_one.sh &

export FULLDATE="2022/09/15"
# export MELTLEVEL="4.7"
./run_one.sh &

export FULLDATE="2022/09/17"
# export MELTLEVEL="4.8"
./run_one.sh &