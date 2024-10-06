import cv2
import numpy as np


def grabcut(original_image: cv2.typing.MatLike, rect: cv2.typing.Rect):
    number_of_iterations = 5
    height, width, _ = np.shape(original_image)
    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    final_mask, _, __ = cv2.grabCut(
        img=original_image,
        mask=mask,
        rect=rect,
        bgdModel=background_model,
        fgdModel=foreground_model,
        iterCount=number_of_iterations,
        mode=cv2.GC_INIT_WITH_RECT,
    )

    grabcut_mask = np.where(
        (final_mask == cv2.GC_PR_BGD) | (final_mask == cv2.GC_BGD), 0, 1
    ).astype("uint8")
    segmented_image = original_image.copy() * grabcut_mask[:, :, np.newaxis]

    return segmented_image
