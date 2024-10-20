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
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        # self.mask2 = np.zeros(self.img.shape[:2], np.uint8)
        
    def set_rect(self, rect):
        self.rect = rect
    
    def grabcut(self, type):
        # print(self.mask.shape, self.rect, self.img.shape)
        
        fModel = np.zeros((1, 65), np.float64)
        bModel = np.zeros((1, 65), np.float64)
        
        mode = cv2.GC_INIT_WITH_RECT if type == 0 else cv2.GC_INIT_WITH_MASK
        self.mask, bModel, fModel = cv2.grabCut(self.img, self.mask, self.rect if type == 0 else None , bModel, fModel, 1, mode=mode)
        mask3 = np.where((self.mask==cv2.GC_FGD)|(self.mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

        new_img = cv2.bitwise_and(self.img, self.img, mask=mask3)
        return new_img
    def quadratic_bezier(self, p0, p1, p2, t):
        return (1-t)**2 * np.array(p0) + 2 * (1-t) * t * np.array(p1) + t**2 * np.array(p2)

    def draw_quad(self, p0, p1, p2, color, thickness=1):
        for i in np.linspace(0, 1, 10):
            poo = self.quadratic_bezier(p0, p1, p2, i)
            poo = tuple(map(int, poo))
            self.mask = cv2.circle(self.mask, poo, thickness, color, -1)
    
    def path(self, command, cc):
        current_pos = None
        for comman in command:
            if comman[0] == 'M':
                current_pos = comman[1:]
            elif comman[0] == 'Q':
                p0 = current_pos
                p1 = comman[1:3]
                p2 = comman[3:]
                self.draw_quad(p0, p1, p2, cc, 3)
                current_pos = p2
            elif comman[0] == 'L':
                p0 = current_pos
                p1 = comman[1:]
                self.mask = cv2.line(self.mask, tuple(map(int, p0)), tuple(map(int, p1)), cc, 3)
                current_pos = p1
    
    