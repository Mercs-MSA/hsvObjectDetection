"""
Auto-refreshing json setting storage
"""

import atexit
import json
import typing

import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class MainDataStorage:
    """
    Data Storage Class
    """
    def __init__(self) -> None:
        self.use_dual_color_targets = False

        # HSV and SV range values for the first range
        self.hsv_min1 = np.array([0, 0, 0])
        self.hsv_max1 = np.array([0, 0, 0])
        self.sv_min1 = np.array([0, 0, 0])
        self.sv_max1 = np.array([0, 0, 0])

        # HSV and SV range values for the second range
        self.hsv_min2 = np.array([0, 0, 0])
        self.hsv_max2 = np.array([0, 0, 0])
        self.sv_min2 = np.array([0, 0, 0])
        self.sv_max2 = np.array([0, 0, 0])

        self.kernel_size = [0, 0]
        self.min_object_area = 0

        self.cube_width = 0


class CamDataStorage:
    """
    Data Storage Class for Camera
    """
    def __init__(self) -> None:
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.dist = [0, 0, 0, 0, 0]


class MainSettingsHandler(FileSystemEventHandler):
    """
    File watchdog
    """
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        """
        Callback for when files are changed
        """

        if not event.is_directory and event.src_path.endswith('.json'):
            self.callback()


storage = MainDataStorage()
cam_storage = CamDataStorage()


def load_settings() -> typing.Tuple[dict, dict]:
    """Load and return settings data

    Returns:
        dict: Settings json
    """
    with open('calibrations.json', 'r', encoding="UTF-8") as file:
        settings = json.load(file)

    with open('camera_calib.json', 'r', encoding="UTF-8") as file:
        camera_calib = json.load(file)

    return settings, camera_calib


def update_settings():
    """
    Update settings in DataStorage class
    """

    settings, camera_calib = load_settings()
    print("Settings updated:", settings)

    storage.use_dual_color_targets = settings["USE_DUAL_COLOR_OBJECT"]

    # HSV and SV range values for the first range
    storage.hsv_min1 = np.array(settings["color_ranges"][0]["hsv_min"])
    storage.hsv_max1 = np.array(settings["color_ranges"][0]["hsv_max"])
    storage.sv_min1 = np.array(settings["color_ranges"][0]["sv_min"])
    storage.sv_max1 = np.array(settings["color_ranges"][0]["sv_max"])

    # HSV and SV range values for the second range
    storage.hsv_min2 = np.array(settings["color_ranges"][1]["hsv_min"])
    storage.hsv_max2 = np.array(settings["color_ranges"][1]["hsv_max"])
    storage.sv_min2 = np.array(settings["color_ranges"][1]["sv_min"])
    storage.sv_max2 = np.array(settings["color_ranges"][1]["sv_max"])

    storage.kernel_size = settings["kernel_size"]
    storage.min_object_area = settings["min_object_area"]

    storage.cube_width = settings["cube_width"]

    cam_storage.fx = camera_calib["fx"]
    cam_storage.fy = camera_calib["fy"]
    cam_storage.cx = camera_calib["cx"]
    cam_storage.cy = camera_calib["cy"]
    cam_storage.dist = camera_calib["dist"]


update_settings()

event_handler = MainSettingsHandler(update_settings)
observer = Observer()
observer.schedule(event_handler, path='.', recursive=False)
observer.start()

atexit.register(observer.stop)
