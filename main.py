"""
FRC Off-Season Vision
"""

import logging
logging.basicConfig(level=logging.DEBUG)

import platform
import traceback
import pprint
import sys

import cv2
from networktables import NetworkTables

from data_storage import cam_storage, storage
from pipelines import SingleColorPipeline
from settings import CAMERA_ID


__version__ = "0.1.0"


if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)


def main():
    """
    Main OpenCV Loop
    """

    # Create a VideoCapture object to access the webcam
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW) # Use dshow to improve fps on windows
    else:
        cap = cv2.VideoCapture(CAMERA_ID) # linux just works

    pipeline = SingleColorPipeline(storage, cam_storage)

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                logging.error("Failed to capture video from the webcam")
                continue

            _, frame, data = pipeline.run(frame)
            visual, mask = pipeline.get_debug_mats()

            pprint.pprint(data)
            nt.putString("note_pipeline", str(data))

            cv2.imshow("Original with Rectangles", visual)
            cv2.imshow("HSV Mask", mask)

            nt.putBoolean("vision_ok", True)

            # Exit the loop when the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Traceback during pipeline update {e}")
            nt.putBoolean("vision_ok", False)

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ip = sys.argv[1]
    NetworkTables.initialize(server=ip)
    nt = NetworkTables.getTable("Vision")
    # nt.putString("version", __version__)
    main()
