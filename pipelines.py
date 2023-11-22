"""
Vision Pipelines
"""

import typing

import cv2
import cv2.typing
import numpy as np

from utils import imfill
from data_storage import MainDataStorage, CamDataStorage


class ConeCubePipeline:
    """
    Pipeline for detecting cubes and cones
    """
    def __init__(self, storage: MainDataStorage, cam_storage: CamDataStorage) -> None:
        self.cone_mask = None
        self.cube_mask = None
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
            dict: Cube and Cone Pos Data
        """
        self.visual = in_frame.copy()
        data = {"cones": [], "cubes": []}

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2HSV)

        # Create HSV and SV masks using the specified ranges
        cube_mask1 = cv2.inRange(hsv_frame, self.storage.hsv_min1, self.storage.hsv_max1)
        cone_mask1 = cv2.inRange(hsv_frame, self.storage.hsv_min2, self.storage.hsv_max2)

        if self.storage.use_dual_color_targets:
            cube_mask2 = cv2.inRange(hsv_frame, self.storage.sv_min1, self.storage.sv_max1)
            cone_mask2 = cv2.inRange(hsv_frame, self.storage.sv_min2, self.storage.sv_max2)

            # Combine the masks
            combined_cube_mask = cv2.bitwise_or(cube_mask1, cube_mask2)
            combined_cone_mask = cv2.bitwise_or(cone_mask1, cone_mask2)
        else:
            combined_cube_mask = cube_mask1
            combined_cone_mask = cone_mask1

        self.cone_mask = combined_cone_mask
        self.cube_mask = combined_cube_mask

        combined_cube_mask = imfill(combined_cube_mask)
        combined_cone_mask = imfill(combined_cone_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.storage.kernel_size)

        # Cube Morph
        thresh_cube = cv2.threshold(combined_cube_mask, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        morph_cube = cv2.morphologyEx(thresh_cube, cv2.MORPH_CLOSE, kernel)

        # Cone Morph
        thresh_cone = cv2.threshold(combined_cone_mask, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        morph_cone = cv2.morphologyEx(thresh_cone, cv2.MORPH_CLOSE, kernel)

        # Find contours in the combined masks
        contours_cube, _ = cv2.findContours(morph_cube, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_cone, _ = cv2.findContours(morph_cone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours_cube = []
        for cnt in contours_cube:
            if cv2.contourArea(cnt) > self.storage.min_object_area:
                filtered_contours_cube.append(cnt)

        filtered_contours_cone = []
        for cnt in contours_cone:
            if cv2.contourArea(cnt) > self.storage.min_object_area:
                filtered_contours_cone.append(cnt)

        # Draw rectangles around the largest detected contours (if any) for both ranges
        # cv2.drawContours(self.visual, contours_cube, -1, (0, 0, 255))
        for cnt in filtered_contours_cube:
            x, y, w, h = cv2.boundingRect(cnt)

            # Calculate the distance to the cube
            apparent_width = w
            focal_length = self.cam_storage.fx  # Assuming focal length along the x-axis

            distance = (self.storage.cube_width * focal_length) / apparent_width

            cv2.putText(self.visual, f"Cube d={distance:.2f}cm", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(self.visual, (x, y), (x + w, y + h), (0, 255, 0), 2)

            data["cubes"].append({"bounding_box": cv2.boundingRect(cnt),
                                  "distance": round(distance, 5)})

        for cnt in filtered_contours_cone:
            p = cv2.arcLength(cnt, True)
            a = cv2.contourArea(cnt)

            # https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)#Circularity

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.putText(self.visual, f"Cone c={round((4 * np.pi * a) / p ** 2, 2)}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(self.visual, (x, y), (x + w, y + h), (0, 0, 255), 2)

            data["cubes"].append({"bounding_box": cv2.boundingRect(cnt), "perimeter": p, "area": a})

        return True, in_frame, data

    def get_debug_mats(self) -> typing.Tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]:
        """
        Return OpenCV Images for debugging
        """
        return self.visual, self.cone_mask, self.cube_mask
