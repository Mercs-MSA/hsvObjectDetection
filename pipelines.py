"""
Vision Pipelines
"""

import typing

import cv2
import cv2.typing
import numpy as np
import math

from utils import imfill
from data_storage import MainDataStorage, CamDataStorage


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang


class NullPipeline:
    """
    Blank Pipeline class
    """
    def run():
        pass

    def get_debug_mats():
        pass


class SingleColorPipeline(NullPipeline):
    """
    Pipeline for detecting single colored blobs
    """
    def __init__(self, storage: MainDataStorage, cam_storage: CamDataStorage) -> None:
        self.mask = None
        self.visual = None
        self.storage = storage
        self.cam_storage = cam_storage

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
        data = {"objects": []}

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2HSV)

        # Create HSV and SV masks using the specified ranges
        self.mask = cv2.inRange(hsv_frame, self.storage.hsv_min1, self.storage.hsv_max1)

        self.mask = imfill(self.mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.storage.kernel_size)

        # Cone Morph
        thresh_cone = cv2.threshold(self.mask, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        morph_cone = cv2.morphologyEx(thresh_cone, cv2.MORPH_CLOSE, kernel)

        # Find contours in the combined masks
        contours_cone, _ = cv2.findContours(morph_cone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours_cone = []
        for cnt in contours_cone:
            if cv2.contourArea(cnt) > self.storage.min_object_area:
                filtered_contours_cone.append(cnt)

        for idx, cnt in enumerate(filtered_contours_cone):
            p = cv2.arcLength(cnt, True)
            a = cv2.contourArea(cnt)

            # https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity

            x, y, w, h = cv2.boundingRect(cnt)
            c = (x + w // 2, y + h // 2)
            angle = angle3pt(c, (self.mask.shape[1] // 2, self.mask.shape[0]), (self.mask.shape[1] // 2, self.mask.shape[0] - 1))

            cv2.putText(self.visual, f"T{idx}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(self.visual, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawMarker(self.visual, c, (0, 255, 0))

            data["objects"].append({"bounding_box": cv2.boundingRect(cnt), "center": c, "perimeter": p, 
                                    "area": a, "index": idx, "angle": angle})

        return True, in_frame, data

    def get_debug_mats(self) -> typing.Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        """
        Return OpenCV Images for debugging
        """
        return self.visual, self.mask