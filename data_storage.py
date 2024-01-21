"""
Auto-refreshing json setting storage
"""

import atexit
import json
import typing
import logging
import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

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

        logging.info(f"Loaded storage class {self}")

class PipeStorageProvider:
    def __init__(self, pipeline_id: str, default_data: dict) -> bool:
        self.duplicate = False

        for provider in storage_providers:
            if provider.pipeline_id == pipeline_id:
                logging.error(f"Multiple storage providers with the same id exist. {pipeline_id}")
                self.duplicate = True
                return

        storage_providers.append(self)
        self.default_data = default_data
        self.pipeline_id = pipeline_id
        self.data = None

        os.makedirs(f"pipeline_storage/{pipeline_id}", exist_ok=True) # create storage for pipeline
        self.path = f"pipeline_storage/{pipeline_id}/storage.json"

        self.update()


    def update(self):
        if not self.duplicate:
            if (not os.path.exists(self.path)) or os.stat(self.path).st_size == 0:
                with open(self.path, "w") as file:
                    file.write(json.dumps(self.default_data))
            
            self.data = json.load(open(self.path, "r"))
        else:
            logging.critical(f"Couldn't save pipeline config. Another pipeline SotrageProvider exists with the same id.")
            self.data = {}
        


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
            self.callback(event.src_path)


storage_providers: list[PipeStorageProvider] = []

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


def update_settings(path):
    """
    Update settings of sotrage providers
    """

    for provider in storage_providers:
        if os.path.abspath(provider.path) == os.path.abspath(path):
            provider.update()


event_handler = MainSettingsHandler(update_settings)
observer = Observer()
observer.schedule(event_handler, path='.', recursive=True)
observer.start()

atexit.register(observer.stop)
