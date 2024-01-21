"""
FRC Off-Season Vision
"""

import logging

import platform
import traceback
import pprint
import sys
import time

import cv2
from networktables import NetworkTables, NetworkTable
from pipelines import SingleColorPipeline
import data_storage


__version__ = "0.1.0"


def _loop(nt: NetworkTable, storage: data_storage.ApplicationStorageProvider) -> None:
    """
    Main OpenCV Loop
    """

    # Create a VideoCapture object to access the webcam
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(storage.data["camera_id"], cv2.CAP_DSHOW) # Use dshow to improve fps on windows
    else:
        cap = cv2.VideoCapture(storage.data["camera_id"]) # linux just works

    pipeline = SingleColorPipeline(id="NoteDetect")

    last_frame_timestamp = time.time()


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

            fps = 1 / (time.time() - last_frame_timestamp)
            last_frame_timestamp = time.time()

            nt.putNumber("fps", round(fps, 1))

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

def init(ip: str) -> None:
    storage = data_storage.ApplicationStorageProvider({"camera_id": 0})

    NetworkTables.initialize(server=ip)
    nt = NetworkTables.getTable("Vision")
    # nt.putString("version", __version__)
    _loop(nt, storage)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


    if len(sys.argv) != 2:
        print("Error: specify an IP to connect to!")
        sys.exit(0)

    init(sys.argv[1])
