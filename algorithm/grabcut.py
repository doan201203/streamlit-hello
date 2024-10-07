import cv2
import numpy as np


def grabcut(img, rect):
    mask = np.zeros(img.shape[:2], np.uint8)
    bModel = np.zeros((1, 65), np.float64)
    fModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bModel, fModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img 
