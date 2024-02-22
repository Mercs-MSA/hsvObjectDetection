"""
Vision Pipelines
"""

import typing
import time

import cv2
import cv2.typing
import numpy as np

from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from pycoral.adapters.detect import Object as PyCObject

from utils import imfill, angle3pt
import data_storage


class NullPipeline:
    """
    Blank Pipeline class
    """
    def __init__(self):
        self.default_data = {}

    def run(self, in_frame: cv2.typing.MatLike):
        """Run a blank OpenCv Pipeline

        Args:
            in_frame (cv2.typing.MatLike): Input from camera or image source
        """

    def get_debug_mats(self):
        """
        Retrieve debug data from OpenCv pipeline (placeholder)
        """


class SingleColorPipeline(NullPipeline):
    """
    Pipeline for detecting single colored blobs
    """
    def __init__(self, pipe_id: str = "SingleColor") -> None:
        super(SingleColorPipeline, self).__init__()
        self.default_data = \
        {
            "color_range": {
                "hsv_min": [8, 180, 20],
                "hsv_max": [20, 255, 240]
            },
            "kernel_size": [1, 1],
            "min_object_area": 1000
        }

        self.mask = None
        self.visual = None
        self.id = pipe_id
        self.storage = data_storage.PipeStorageProvider(pipe_id, self.default_data)

    def run(self, in_frame: cv2.typing.MatLike) -> typing.Union[bool, cv2.typing.MatLike, dict]:
        """Pipeline

        Args:
            in_frame (cv2.typing.MatLike): Input from camera

        Returns:
            bool: Success
            cv2.typing.MatLike: Input from camera
            dict: Pos Data
        """
        self.visual = in_frame.copy()
        data = {"objects": [], "best": None}

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2HSV)

        # Create HSV and SV masks using the specified ranges
        self.mask = cv2.inRange(hsv_frame, np.array(self.storage.data["color_range"]["hsv_min"]),
                                np.array(self.storage.data["color_range"]["hsv_max"]))

        self.mask = imfill(self.mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.storage.data["kernel_size"])

        # Morph
        thresh = cv2.threshold(self.mask, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours in the combined masks
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > self.storage.data["min_object_area"]:
                filtered_contours.append(cnt)

        for idx, cnt in enumerate(filtered_contours):
            p = cv2.arcLength(cnt, True)
            a = cv2.contourArea(cnt)

            # https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity

            x, y, w, h = cv2.boundingRect(cnt)
            c = (x + w // 2, y + h // 2)
            angle = angle3pt(c, (self.mask.shape[1] // 2, self.mask.shape[0]),
                             (self.mask.shape[1] // 2, self.mask.shape[0] - 1))

            cv2.putText(self.visual, f"T{idx}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(self.visual, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawMarker(self.visual, c, (0, 255, 0))

            data["objects"].append({"bounding_box": cv2.boundingRect(cnt),
                                    "center": c, "perimeter": p,
                                    "area": a, "index": idx, "yaw": angle})

        data["best"] = max(data["objects"], key=lambda x: x["area"], default=None)

        return True, in_frame, data

    def get_debug_mats(self) -> typing.Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        """
        Return OpenCV Images for debugging
        """
        return self.visual, self.mask


class PyCoralObjectReturn(typing.TypedDict):
    """
    Dict return type for single PyCoralPipeline object
    """
    bounding_box: tuple[int, int, int, int]
    center: float
    perimeter: int
    area: int
    index: int
    yaw: float
    accuracy: float


class PyCoralPipeReturnData(typing.TypedDict):
    """
    Dict return type for PyCoralPipeline
    """
    best: typing.Union[PyCoralObjectReturn, dict[None]]
    objects: list[PyCoralObjectReturn]


class PyCoralPipeline(NullPipeline):
    """
    Pipeline for detecting single colored blobs
    """
    def __init__(self, pipe_id: str = "SingleColor") -> None:
        super(PyCoralPipeline, self).__init__()
        self.default_data = \
        {
            "model": "edgetpu.tflite",
            "labels": "labels.txt",
            "min_object_area": 1000,
            "min_threshold": 0.5
        }

        self.visual = None
        self.id = pipe_id
        self.storage = data_storage.PipeStorageProvider(pipe_id, self.default_data)

        self.labels = read_label_file(self.storage.data["labels"])
        self.interpreter = make_interpreter(self.storage.data["model"])
        self.interpreter.allocate_tensors()

    def run(self, in_frame: cv2.typing.MatLike) -> typing.Tuple[bool, cv2.typing.MatLike, PyCoralPipeReturnData]:
        """Pipeline

        Args:
            in_frame (cv2.typing.MatLike): Input from camera

        Returns:
            bool: Success
            cv2.typing.MatLike: Input from camera
            dict: Pos Data
        """
        self.visual = in_frame.copy()

        data: PyCoralPipeReturnData = {"objects": [], "best": {}}

        image = Image.fromarray(cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB))
        _, scale = common.set_resized_input(
            self.interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(self.interpreter, self.storage.data["min_threshold"], scale)


        filtered_contours: list[PyCObject] = []
        for obj in objs:
            if obj.bbox.area > self.storage.data["min_object_area"]:
                filtered_contours.append(obj)

        for idx, obj in enumerate(filtered_contours):
            p = (2 * obj.bbox.width) + (2 * obj.bbox.height)
            a = obj.bbox.area

            # https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity

            x, y, w, h = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.width, obj.bbox.height]
            c = (x + w // 2, y + h // 2)
            angle = angle3pt(c, (in_frame.shape[1] // 2, in_frame.shape[0]),
                             (in_frame.shape[1] // 2, in_frame.shape[0] - 1))

            cv2.putText(self.visual, f"T{idx}; {self.labels.get(obj.id, obj.id)}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(self.visual, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawMarker(self.visual, c, (0, 255, 0))

            data["objects"].append({"bounding_box": (x, y, w, h),
                                    "center": c, "perimeter": p,
                                    "area": a, "index": idx, "yaw": angle,
                                    "accuracy": obj.score})

        cv2.putText(self.visual, f"{1 / (inference_time * 1000) * 1000:.1f} FPS",
                    (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0))

        data["best"] = max(data["objects"], key=lambda x: x["area"], default=None)

        return True, in_frame, data

    def get_debug_mats(self) -> typing.Tuple[cv2.typing.MatLike]:
        """
        Return OpenCV Images for debugging
        """
        return self.visual


class PyCoralPoseObjectReturn(typing.TypedDict):
    """
    Dict return type for single PyCoralPosePipeline object
    """
    bounding_box: tuple[int, int, int, int]
    center: float
    perimeter: int
    area: int
    index: int
    yaw: float
    distance: float
    accuracy: float


class PyCoralPosePipeReturnData(typing.TypedDict):
    """
    Dict return type for PyCoralPosePipeline
    """
    best: typing.Union[PyCoralPoseObjectReturn, dict[None]]
    objects: list[PyCoralPoseObjectReturn]


class PyCoralPosePipeline(NullPipeline):
    """
    Pipeline for detecting pose of note via pycoral tpu runtime
    """
    def __init__(self, app_storage: data_storage.ApplicationStorageProvider, pipe_id: str = "PyCoralPoser") -> None:
        super(PyCoralPosePipeline, self).__init__()
        self.default_data = \
        {
            "model": "edgetpu.tflite",
            "labels": "labels.txt",
            "min_object_area": 1000,
            "min_threshold": 0.5,
            "note_width": 14
        }

        self.visual = None
        self.id = pipe_id
        self.storage = data_storage.PipeStorageProvider(pipe_id, self.default_data)

        self.fx = app_storage.data["camera_calib"]["fx"]
        self.fy = app_storage.data["camera_calib"]["fy"]
        self.cx = app_storage.data["camera_calib"]["cx"]
        self.cy = app_storage.data["camera_calib"]["cy"]
        self.dist_coeffs = np.array(app_storage.data["camera_calib"]["dist_coeff"])

        self.labels = read_label_file(self.storage.data["labels"])
        self.interpreter = make_interpreter(self.storage.data["model"])
        self.interpreter.allocate_tensors()

    def run(self, in_frame: cv2.typing.MatLike) -> typing.Tuple[bool, cv2.typing.MatLike, PyCoralPosePipeReturnData]:
        """Pipeline

        Args:
            in_frame (cv2.typing.MatLike): Input from camera

        Returns:
            bool: Success
            cv2.typing.MatLike: Input from camera
            dict: Pos Data
        """
        self.visual = in_frame.copy()

        data: PyCoralPosePipeReturnData = {"objects": [], "best": {}}

        image = Image.fromarray(cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB))
        _, scale = common.set_resized_input(
            self.interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(self.interpreter, self.storage.data["min_threshold"], scale)


        filtered_contours: list[PyCObject] = []
        for obj in objs:
            if obj.bbox.area > self.storage.data["min_object_area"]:
                filtered_contours.append(obj)

        for idx, obj in enumerate(filtered_contours):
            p = (2 * obj.bbox.width) + (2 * obj.bbox.height)
            a = obj.bbox.area

            # https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity

            x, y, w, h = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.width, obj.bbox.height]
            c = (x + w // 2, y + h // 2)
            d = (self.storage.data["note_width"] * self.fx) / obj.bbox.width
            angle = angle3pt(c, (in_frame.shape[1] // 2, in_frame.shape[0]),
                             (in_frame.shape[1] // 2, in_frame.shape[0] - 1))

            cv2.putText(self.visual, f"T{idx}; {d:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.rectangle(self.visual, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawMarker(self.visual, c, (0, 255, 0))

            data["objects"].append({"bounding_box": (x, y, w, h),
                                    "center": c, "perimeter": p,
                                    "area": a, "index": idx,
                                    "yaw": angle, "distance": d,
                                    "accuracy": obj.score})

        cv2.putText(self.visual, f"{1 / (inference_time * 1000) * 1000:.1f} FPS",
                    (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0))

        data["best"] = max(data["objects"], key=lambda x: x["area"], default=None)

        return True, in_frame, data

    def get_debug_mats(self) -> typing.Tuple[cv2.typing.MatLike]:
        """
        Return OpenCV Images for debugging
        """
        return self.visual
