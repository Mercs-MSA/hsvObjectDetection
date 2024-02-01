"""
FRC 6369 Vision
"""

import logging

import platform
import traceback
import pprint
import time
import json

import cv2
import ntcore
from pipelines import PyCoralPipeline
import data_storage


__version__ = "0.1.0"


def _loop(nt: ntcore.NetworkTable, storage: data_storage.ApplicationStorageProvider) -> None:
    """
    Main OpenCV Loop
    """

    # Create a VideoCapture object to access the webcam
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(storage.data["camera_id"],
                               cv2.CAP_DSHOW) # Use dshow to improve fps on windows
    else:
        cap = cv2.VideoCapture(storage.data["camera_id"]) # linux just works

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, storage.data["camera_resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, storage.data["camera_resolution"][1])

    if storage.data["camera_exp"]:
        cap.set(cv2.CAP_PROP_EXPOSURE, storage.data["camera_exp"])

    pipeline = PyCoralPipeline(pipe_id="NoteDetect")

    last_frame_timestamp = time.time()

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                logging.error("Failed to capture video from the webcam")
                continue

            _, frame, data = pipeline.run(frame)
            visual = pipeline.get_debug_mats()

            pprint.pprint(data)
            nt.putString("note_pipeline", json.dumps(data))

            cv2.imshow("Original with Rectangles", visual)

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

def init() -> None:
    """
    Initialize pipeline processing
    """
    storage = data_storage.ApplicationStorageProvider({"camera_id": 0,
                                                       "camera_resolution": [1280, 720],
                                                       "camera_exp": None, "nt_version": 4,
                                                       "nt_address": "localhost"})

    inst = ntcore.NetworkTableInstance.getDefault()
    inst.setServer(storage.data["nt_address"])
    client_version = storage.data["nt_version"]
    if client_version == 3:
        inst.startClient3("vision")
    elif client_version == 4:
        inst.startClient4("vision")
    else:
        logging.warning("Client version must either be 3 or 4. Defaulting to v4")
        inst.startClient4("vision")

    nt = inst.getTable("Vision")
    nt.putString("version", __version__)

    _loop(nt, storage)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    init()
