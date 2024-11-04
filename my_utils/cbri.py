from my_utils.superpoint import SuperPointFrontend
import faiss
import numpy as np
from scipy.cluster.vq import vq
import cv2
from numpy.linalg import norm
import pickle

class CBRI:
    def __init__(self):
        self.detector = SuperPointFrontend(weights_path='./my_utils/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
        with open('./my_utils/kmean.pkl', 'rb') as f:
          self.kmeans = pickle.load(f)

        with open('./my_utils/idf.pkl', 'rb') as f:
          self.idf = pickle.load(f)
          
        with open('./my_utils/tfidf.pkl', 'rb') as f:
          self.tfidf = pickle.load(f)
        
        with open('./my_utils/db.pkl', 'rb') as f:
          self.db = pickle.load(f)
        
    def embed(self, img):
      if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
      kp, desc, heat = self.detector.run(img.astype('float32') / 255.)
      # self.desc = desc.T
      img_visual_words, dis = vq(desc.T, self.kmeans)
      emm = np.zeros(256)
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