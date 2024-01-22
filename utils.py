"""
OpenCV Utilities
"""

import math

import cv2
import cv2.typing
import numpy as np


def imfill(input_mask: cv2.typing.MatLike):
    """Fill in mask contours

    Args:
        input_mask (cv2.typing.MatLike): Mask to work with

    Returns:
        cv2.typing.MatLike: Mask with filled in contours
    """

    # Find contours in the input mask
    contours, _ = cv2.findContours(input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask to fill the blobs
    filled_mask = np.zeros_like(input_mask)

    # Draw filled contours on the mask
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    return filled_mask


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang
