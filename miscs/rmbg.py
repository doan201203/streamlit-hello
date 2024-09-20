import cv2 as cv 
import numpy as np

img = cv.imread('../datasets/biensoxe/test/labels/2xemay142.png')
img[np.where(img[:, :, 2] == 170)] = [0, 0, 0]
# cv.save('test.png', img)
cv.imwrite('../datasets/biensoxe/test/labels/2xemay142.png', img)
print(np.unique(img))
