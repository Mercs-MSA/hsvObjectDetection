"""
FRC Off-Season Vision
"""

import platform
import pprint
import sys

import cv2
from networktables import NetworkTables

from data_storage import cam_storage, storage
from pipelines import ConeCubePipeline
from settings import CAMERA_ID

import logging

__version__ = "0.1.0"

logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)


def main():
    """
    Main OpenCV Loop
    """

    # Create a VideoCapture object to access the webcam
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_ID)

    pipeline = ConeCubePipeline(storage, cam_storage)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture video from the webcam.")
            break

        _, frame, data = pipeline.run(frame)
        visual, cone_mask, cube_mask = pipeline.get_debug_mats()

        pprint.pprint(data)

        # Display the original frame with rectangles, HSV masks for both ranges
        cv2.imshow("Original with Rectangles", visual)
        cv2.imshow("Cube HSV Mask", cube_mask)
        cv2.imshow("Cone HSV Mask", cone_mask)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ip = sys.argv[1]
    NetworkTables.initialize(server=ip)
    nt = NetworkTables.getTable("Vision")
    nt.putString("version", __version__)
    main()
