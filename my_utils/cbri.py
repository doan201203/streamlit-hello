from my_utils.superpoint import SuperPointFrontend
import faiss
import numpy as np
from scipy.cluster.vq import vq
import cv2
from numpy.linalg import norm
import pickle
import os

class CBRI:
    def __init__(self):
        self.detector = SuperPointFrontend(weights_path='./my_utils/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
        self.db = self.docf()
        print(self.db.shape)
        
        with open('./my_utils/kmean.pkl', 'rb') as f:
          self.kmeans = pickle.load(f)

        with open('./my_utils/idf.pkl', 'rb') as f:
          self.idf = pickle.load(f)
          
        with open('./my_utils/tfidf.pkl', 'rb') as f:
          self.tfidf = pickle.load(f)
        
    
    def docf(self):
      wd = './datasets/coco17'
      img = []
      print(sorted(os.listdir(wd), key=lambda x: int(x.split('.')[0])))
      for x in sorted(os.listdir(wd), key=lambda x: int(x.split('.')[0])):
        file = os.path.join(wd, x)
        with open(file, 'rb') as f:
          arr_img = pickle.load(f)
        for i in arr_img:
          img.append(i)
      return np.asarray(img, dtype=np.ndarray)
    
    def embed(self, img):
      if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
      kp, desc, heat = self.detector.run(img / np.float32(255))
      # self.desc = desc.T
      img_visual_words, dis = vq(desc.T, self.kmeans)
      emm = np.zeros(512)
      for w in img_visual_words:
        emm[w] += 1
      
      return emm * self.idf
    
    def top_k(self, que, k):
      emb = self.embed(que)
      b = self.tfidf
      cos = np.dot(emb, b.T) / (norm(emb) * norm(b, axis=1))
      idx = np.argsort(-cos)[:k]
      conf = cos[idx]
      return idx, conf