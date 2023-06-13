#!/bin/bash

# Check if the camera is connected by looking for a specific device (e.g., /dev/video0)
CAMERA_DEVICE="/dev/video0"

# Check if the camera device exists
if [ -e "$CAMERA_DEVICE" ]; then
    echo "Camera detected. Launching face_detection.py..."

    # Run the face_detection.py script using the current Python environment
    python3 face_detection.py

else
    echo "Warning: Camera not detected. Installing v4l-utils..."

    # Update package list and install v4l-utils to check camera devices
    sudo apt update
    sudo apt install -y v4l-utils

    # Check again if the camera is detected after installation
    if [ -e "$CAMERA_DEVICE" ]; then
        echo "Camera detected after installing v4l-utils. Launching face_detection.py..."

        # Run the face_detection.py script using the current Python environment
        python3 face_detection.py
    else
        echo "Still no camera detected. Please check the connection manually."
    fi
fi