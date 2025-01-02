#!/bin/bash

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "Error: DISPLAY variable is not set" >&2
    exit 1
fi

echo "starting tint2 on display $DISPLAY ..."

# Create config directory if it doesn't exist
mkdir -p $HOME/.config/tint2

# Copy our tint2rc if it doesn't exist
if [ ! -f $HOME/.config/tint2/tint2rc ]; then
    cp $(dirname $0)/tint2rc $HOME/.config/tint2/tint2rc
fi

# Start tint2 and capture its stderr
tint2 2>/tmp/tint2_stderr.log &

# Wait for tint2 window properties to appear
timeout=30
while [ $timeout -gt 0 ]; do
    if xdotool search --class "tint2" >/dev/null 2>&1; then
        break
    fi
    sleep 1
    ((timeout--))
done

if [ $timeout -eq 0 ]; then
    echo "tint2 stderr output:" >&2
    cat /tmp/tint2_stderr.log >&2
    exit 1
fi

# Remove the temporary stderr log file
rm /tmp/tint2_stderr.log
