#!/bin/bash

# Try to open the camera using OpenCV (camera 0 is usually the default)
python3 -c "
import cv2

cap = cv2.VideoCapture(1)  # 0 is the default camera device
if not cap.isOpened():
    print('Error: Camera not found or cannot be accessed.')
    exit(1)

print('Camera detected. Launching the application...')
cap.release()
"

# If the camera is detected, launch your face_detection.py script
echo "Launching face_detection.py..."
python3 ./face_detection.py
