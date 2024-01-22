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
    """
    Storage provider for single-pipeline storage
    """
    def __init__(self, pipeline_id: str, default_data: dict) -> None:
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


    def update(self) -> None:
        """Refresh data
        """
        if not self.duplicate:
            if (not os.path.exists(self.path)) or os.stat(self.path).st_size == 0:
                with open(self.path, "w", encoding="utf-8") as file:
                    file.write(json.dumps(self.default_data, indent=4))

            self.data = json.load(open(self.path, "r", encoding="utf-8"))
        else:
            logging.critical("Couldn't update pipeline config. \
                             Another pipeline provider exists with the same id.")
            self.data = {}


class ApplicationStorageProvider:
    """
    Storage provider for whole-app storage
    """
    def __init__(self, default_data: dict, app_id: str = "app") -> None:
        self.duplicate = False

        for provider in storage_providers:
            if provider.pipeline_id == app_id:
                logging.error(f"Multiple storage providers with the same id exist. {app_id}")
                self.duplicate = True
                return

        storage_providers.append(self)
        self.default_data = default_data
        self.pipeline_id = app_id
        self.data = None

        os.makedirs(f"app_storage/{app_id}", exist_ok=True) # create storage
        self.path = f"app_storage/{app_id}/storage.json"

        self.update()


    def update(self) -> None:
        """Refresh data
        """
        if not self.duplicate:
            if (not os.path.exists(self.path)) or os.stat(self.path).st_size == 0:
                with open(self.path, "w", encoding="utf-8") as file:
                    file.write(json.dumps(self.default_data, indent=4))

            self.data = json.load(open(self.path, "r", encoding="utf-8"))
        else:
            logging.critical("Couldn't update app config. \
                             Another provider exists with the same id.")
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
