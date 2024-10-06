import numpy as np
import cv2 as cv
import pickle
import xml.etree.ElementTree as ET
import os

class HaarFeature:
  def __init__(self, features_file):
    self.features_file = features_file
    self.X = []
    self.y = []
    self.load_features()
  
  def load_features(self):
    tree = ET.parse(self.features_file)
    root = tree.getroot()
    self.features = []
    fts = root.findall('.//features/_/rects')
    for fe in fts:
      rects = fe.findall('_')
      rects_values = []
      for r in rects:
        x, y, w, h, weight = map(float, r.text.strip().split())
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        rects_values.append((x, y, w, h, weight))
      self.features.append(rects_values)
    
    pos = './datasets/faces_and_non_faces_data/p/'
    neg = './datasets/faces_and_non_faces_data/n/'
    for i in os.listdir(pos):
      img = cv.imread(os.path.join(pos, i))
      if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      itg = cv.integral(img)
      self.X.append(self.extract_ft(itg, 0, 0))
      self.y.append(1)
    for i in os.listdir(neg):
      img = cv.imread(os.path.join(neg, i))
      if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      itg = cv.integral(img)
      self.X.append(self.extract_ft(itg, 0, 0))
      self.y.append(0)
      
    self.X = np.array(self.X)
    self.y = np.array(self.y)


  def extract_ft(self, integral, topx, topy):
    feat = []
    pref = integral
    for feature in self.features:
        re = 0
        for rec in feature:
          i, j, w, h, weight = rec
          re += weight * (pref[topy + j + h][topx + i + w] + pref[topy + j][topx + i] - pref[topy + j][topx + i + w] - pref[topy + j + h][topx + i])
          # re += weight * (pref[j + h][i + w] + pref[j][i] - pref[j][i + w] - pref[j + h][i])
        feat.append(re)
    return np.asarray(feat)
  
  def detect_faces_multiscale(self, image, classifier, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24), stepSize=10, threshold=0.7):
      faces = []
      h, w = image.shape[:2]
      
      scale = 1.0
      while True:
          new_w = int(w * scale)
          new_h = int(h * scale)
          print(scale, (new_w, new_h))
          if new_w < minSize[0] or new_h < minSize[1]:
            break
            
          resized_img = cv.resize(image, (0, 0), fx=scale, fy=scale)
          integral_img = cv.integral(resized_img)
          
          # Di chuyển cửa sổ quét với kích thước cố định minSize
          x, y = 0, 0
          while resized_img.shape[1] - minSize[0] > x:
              while resized_img.shape[0] - minSize[1] > y:
                  features = self.extract_ft(integral_img, x, y)
                  if classifier.predict_proba([features])[0][1] >= threshold:
                      real_x = int(x / scale)
                      real_y = int(y / scale)
                      real_w = int(minSize[0] / scale)
                      real_h = int(minSize[1] / scale)
                      faces.append((real_x, real_y, real_w, real_h))
                  y += stepSize
              x += stepSize
              y = 0
          
          # Giảm kích thước ảnh theo scaleFactor
          scale /= scaleFactor
      
      # Lọc các khuôn mặt bằng cách kiểm tra các neighbors
      tmp = cv.resize(image.copy(), (w, h))
      f = []
      if len(tmp.shape) == 3:
          tmp = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
      
      ptr = len(faces) // 2
      #ptr->last then 0-> ptr-1
      for x, y, w, h in faces[::-1]:
          crop = tmp[y: y + h, x: x + w]
          if np.count_nonzero(crop) < 0.80 * crop.shape[0] * crop.shape[1]:
              continue
          tmp[y: y + h, x: x + w] = 0
          f.append((x, y, w, h))
      return f
