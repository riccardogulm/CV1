import cv2
import numpy as np
import os
import requests
import time
import threading
import queue

# Ensure that the directory exists for saving model files
save_dir = './model_files'
os.makedirs(save_dir, exist_ok=True)

# URLs for the MobileNet SSD model and weights
prototxt_url = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt'
caffemodel_url = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'

# File paths where the model and weights will be saved
prototxt_path = os.path.join(save_dir, 'deploy.prototxt')
caffemodel_path = os.path.join(save_dir, 'mobilenet_iter_73000.caffemodel')

# Download the files if they don't exist
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        response = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

download_file(prototxt_url, prototxt_path)
download_file(caffemodel_url, caffemodel_path)

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Shared queue for passing frames between threads
frame_queue = queue.Queue()

# Thread 1: Capture video from webcam
def capture_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        frame_queue.put(frame)

    cap.release()

# Thread 2: Process captured frames (resize, grayscale)
def process_frame():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            if frame is None or not isinstance(frame, np.ndarray):
                continue  # Skip this frame

            # Resize the image to 300x300
            resized_frame = cv2.resize(frame, (300, 300))

            # Convert the resized image to grayscale
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            # Put the processed frame back in the queue for detection
            frame_queue.put((frame, blurred_frame))

# Thread 3: Detect faces using MobileNet SSD and Haar Cascade
def detect_faces():
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        if not frame_queue.empty():
            item = frame_queue.get()
            if isinstance(item, tuple):
                frame, processed_frame = item
            else:
                frame = item
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h, w = frame.shape[:2]

            # MobileNet SSD face detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            net.setInput(blob)
            detections = net.forward()

            # Haar Cascade face detection
            faces_haar = haar_cascade.detectMultiScale(processed_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Copy frame for visualization
            image_mobilenet = frame.copy()
            image_haar = frame.copy()

            # Draw boxes for MobileNet SSD and rescale coordinates
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(image_mobilenet, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"MobileNet -Confidence: {confidence * 100:.2f}%"
                    cv2.putText(image_mobilenet, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # Draw rectangles for Haar Cascade detection
            for (x, y, w, h) in faces_haar:
                cv2.rectangle(image_haar, (x, y), (x + w, y + h), (255, 0, 0), 2)
                label = "Haar Cascade"
                cv2.putText(image_haar, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Combine images for visualization
            combined_image = np.hstack((image_mobilenet, image_haar))

            # Display the detection results
            cv2.imshow("Detections", combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Main function to start threads
def main():
    # Create and start the threads
    capture_thread = threading.Thread(target=capture_video, daemon=True)
    process_thread = threading.Thread(target=process_frame, daemon=True)
    detection_thread = threading.Thread(target=detect_faces, daemon=True)

    capture_thread.start()
    process_thread.start()
    detection_thread.start()

    # Keep the main thread running to avoid program termination
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating program...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
