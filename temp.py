import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
import time

# Camera intrinsic parameters and distortion coefficients
fx = 288.14
fy = 288.61
cx = 324.93
cy = 215.98
dist_coeffs = np.array([ 0.019, -0.035, 0, 0.001, -0.078, 0.017, 0.025, -0.104 ] )

# Load the Edge TPU model and label map
model_path = "edgetpu.tflite"
label_path = "labels.txt"

interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load label map
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Function to calculate distance based on object size
def calculate_distance(inner_diameter_pixels):
    # Assuming inner diameter of the note is 10 inches
    distance = (10.0 * fx) / inner_diameter_pixels
    return distance

# Open webcam
cap = cv2.VideoCapture("/dev/video0")

interpreter = make_interpreter("edgetpu.tflite")
interpreter.allocate_tensors()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    detected_objects = detect.get_objects(interpreter, 0.5, scale)

    # Display the results on the frame
    for obj in detected_objects:
        ymin, xmin, ymax, xmax = [obj.bbox.ymin, obj.bbox.xmin, obj.bbox.ymax, obj.bbox.xmax]

        label = "note"
        confidence = obj.score

        if confidence > 0.0:  # Display only if confidence is above a certain threshold
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            frame = cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate object size and distance
            inner_diameter_pixels = obj.bbox.width
            distance = calculate_distance(inner_diameter_pixels)
            print(distance)
            cv2.putText(frame, f"Distance: {distance:.2f} inches", (xmin, ymin - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection with Coral TPU', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
