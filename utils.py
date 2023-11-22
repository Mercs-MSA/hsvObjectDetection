"""
OpenCV Utilities
"""

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
