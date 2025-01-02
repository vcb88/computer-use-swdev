#!/bin/bash

set -e

export DISPLAY=:${DISPLAY_NUM}
./xvfb_startup.sh
./mutter_startup.sh
sleep 2  # Give mutter time to initialize
./tint2_startup.sh
./x11vnc_startup.sh
