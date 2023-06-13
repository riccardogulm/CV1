# Use an official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies (opencv-python-headless, numpy, and requests)
RUN pip install --no-cache-dir opencv-python-headless numpy requests

# Copy the necessary files into the container
COPY face_detection.py .
COPY launch_face_detection.sh .

# Give execution permissions to the Bash script
RUN chmod +x launch_face_detection.sh

# Set the default command to run the script
CMD ["bash", "./launch_face_detection.sh"]
