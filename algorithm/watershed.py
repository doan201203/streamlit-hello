import numpy as np
import cv2 as cv

def watershed(img, kernelsize, thres):
  # Load the image
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  
  # Noise removal
  kernel = np.ones((kernelsize, kernelsize), np.uint8)
  opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
  
  # Sure background area
  sure_bg = cv.dilate(opening, kernel)
  
  # Finding sure foreground area
  dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
  ret, sure_fg = cv.threshold(dist_transform, thres * dist_transform.max(), 255, 0)
  
  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv.subtract(sure_bg, sure_fg)
  
  # Marker labelling
  ret, markers = cv.connectedComponents(sure_fg)
  
  # Add one to all labels so that sure background is not 0, but 1
  markers = markers + 1
  
  # Now, mark the region of unknown with zero
  markers[unknown == 255] = 0
  
  markers = cv.watershed(img, markers)
  img[markers == -1] = [255, 0, 0]
  return img, markers

def get_mask(img):
  mask = np.zeros(img.shape, dtype=np.uint8)
  all_labels = np.unique(img)
  
  for label in all_labels:
    if label == -1:
      continue
    xmin, xmax = np.where(img == label)[0].min(), np.where(img == label)[0].max()
    ymin, ymax = np.where(img == label)[1].min(), np.where(img == label)[1].max()
    ratio_x = (xmax - xmin) / img.shape[0]
    ratio_y = (ymax - ymin) / img.shape[1]
    if (ratio_x >= 0.3 and ratio_x <= 0.6) and (ratio_y >= 0 and ratio_y <= 0.3):
      mask[img == label] = 255
    # print(ratio_x, ratio_y)
    # print()
  return mask

def _2dice(pred, gt):
  pred = pred.flatten()
  gt = gt.flatten()
  intersection = np.sum(pred[gt == 255])
  return 2 * intersection / (np.sum(pred) + np.sum(gt))

def iou(pred, gt):
  intersection = np.sum(pred[gt == 255])
  union = np.sum(pred) + np.sum(gt) - intersection
  return intersection / union  