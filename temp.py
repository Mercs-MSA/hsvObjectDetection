import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Camera intrinsic parameters and distortion coefficients
fx = 668.6838165427944
fy = 667.9713968653047
cx = 313.654242908748
cy = 230.16875102271155
dist_coeffs = np.array([0.1741479469487125, -1.2438486729108107, -0.0011594095682407426, -8.395661730281217E-6, 1.9474548124079918])

# Load the Edge TPU model and label map
model_path = "edgetpu.tflite"
label_path = "labelmap.txt"

interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load label map
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Function to calculate distance based on object size
def calculate_distance(inner_diameter_pixels):
    # Assuming inner diameter of the note is 10 inches
    real_inner_diameter = 10.0  # inches
    focal_length = (fx + fy) / 2  # Assuming square pixels
    object_width = real_inner_diameter * fx / inner_diameter_pixels
    distance = (real_inner_diameter * focal_length) / object_width
    return distance

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame to match the input size of the model
    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
    input_size = common.input_size(interpreter)

    # Preserve the aspect ratio of the frame
    h, w, _ = frame.shape
    aspect_ratio = w / h
    new_width = int(input_size * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, input_size))

    # Center-crop the resized frame
    start_col = max(0, (new_width - input_size) // 2)
    resized_frame = resized_frame[:, start_col:start_col + input_size]

    input_tensor.copy_from(np.expand_dims(resized_frame, axis=0))

    # Run inference
    interpreter.invoke()

    # Get results
    output = interpreter.tensor(interpreter.get_output_details()[0]['index']).buffer().raw
    output = np.array(output)
    detected_objects = output.reshape(-1, 6)

    # Display the results on the frame
    for obj in detected_objects:
        ymin, xmin, ymax, xmax = obj[0:4]
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        label_id = int(obj[4])
        label = labels[label_id]
        confidence = obj[5]

        if confidence > 0.5:  # Display only if confidence is above a certain threshold
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate object size and distance
            inner_diameter_pixels = xmax - xmin
            distance = calculate_distance(inner_diameter_pixels)
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
