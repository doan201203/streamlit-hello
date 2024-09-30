import numpy as np

def _2dice(pred, gt):
  pred = pred.flatten()
  gt = gt.flatten()
  intersection = np.sum(pred[gt == 255])
  return 2 * intersection / (np.sum(pred) + np.sum(gt))

def iou(pred, gt):
  intersection = np.sum(pred[gt == 255])
  union = np.sum(pred) + np.sum(gt) - intersection
  return intersection / union  
