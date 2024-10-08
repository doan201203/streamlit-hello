import cv2
import numpy as np


# def grabcut(img, rect):
#     mask = np.zeros(img.shape[:2], np.uint8)
#     bModel = np.zeros((1, 65), np.float64)
#     fModel = np.zeros((1, 65), np.float64)

#     cv2.grabCut(img, mask, rect, bModel, fModel, 5, cv2.GC_INIT_WITH_RECT)

#     mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
#     img = img * mask2[:, :, np.newaxis]
#     return img 

class Grabcut:
    def __init__(self, img) -> None:
        self.img = img
        self.rect = None
        self.copy = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
    
    def set_rect(self, rect):
        self.rect = rect
    
    def grabcut(self, type):
        bModel = np.zeros((1, 65), np.float64)
        fModel = np.zeros((1, 65), np.float64)
        mode = cv2.GC_INIT_WITH_RECT if type == 0 else cv2.GC_INIT_WITH_MASK
        cv2.grabCut(self.img, self.mask, self.rect, bModel, fModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((self.mask==2)|(self.mask==0), 0, 1).astype('uint8')
        img = self.img * mask2[:, :, np.newaxis]
        return img
    
    def draw_tochup_curve(self, points, type):
        if type == 'bg':
            cv2.circle(self.copy, points, 3, (0, 0, 255), -1)
            cv2.circle(self.mask, points, 3, cv2.GC_BGD, -1)
        else:
            cv2.circle(self.copy, points, 3, (255, 0, 0), -1)
            cv2.circle(self.mask, points, 3, cv2.GC_FGD, -1)
    