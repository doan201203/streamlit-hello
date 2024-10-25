import cv2
import numpy as np
from PIL import Image
from my_utils.db_handle import DBHandle
from streamlit.runtime.uploaded_file_manager import UploadedFile
from miscs.face_models.yunet import YuNet
from miscs.face_models.sface import SFace
from google.cloud import firestore
import re, unicodedata

class FaceController:
  def __init__(self, dbname) -> None:
    self.db = DBHandle(dbname)
    self.detector = YuNet(
      modelPath='./miscs/face_models/face_detection_yunet_2023mar.onnx',
      confThreshold=0.85,
    )
    self.regconize = SFace(
      modelPath='./miscs/face_models/face_recognition_sface_2021dec.onnx',
      disType=0,
    )
    
  def detect(self, img: np.ndarray, scale_factor: float = 1.3):
    heigh = img.shape[0]
    width = img.shape[1]
    
    scale = 1.0
    while scale * min(heigh, width) > 24:
      img_resized = cv2.resize(img, (int(width * scale), int(heigh * scale)))
      h, w = img_resized.shape[:2]
      self.detector.setInputSize((w, h))
      bboxs = self.detector.infer(img_resized)
      print(bboxs)
      if len(bboxs) > 0:
        return (bboxs, img_resized)
      scale /= scale_factor
    
    return ([], img)
  
  def insert(self, msv: str, name: str, thesv: UploadedFile, chandung: UploadedFile):
    _thesv_img = Image.open(thesv)
    _chandung_img = Image.open(chandung)
    thesv_img = cv2.cvtColor(np.array(_thesv_img), cv2.COLOR_RGB2BGR)
    chandung_img = cv2.cvtColor(np.array(_chandung_img), cv2.COLOR_RGB2BGR)
    
    bboxs_thesv, thesv_img = self.detect(thesv_img)
    bboxs_chandung, chandung_img = self.detect(chandung_img)
    
    if len(bboxs_thesv) == 0 or len(bboxs_chandung) == 0:
      return -1
    if len(bboxs_thesv) > 1 or len(bboxs_chandung) > 1:
      return -2
    
    chandung_feature = self.regconize.infer(chandung_img, bboxs_chandung[0][:-1])
    thesv_feature = self.regconize.infer(thesv_img, bboxs_thesv[0][:-1])
    
    thesv.seek(0)
    chandung.seek(0)
    self.db.upload_file(thesv, 'face_dataset/TheSV')
    self.db.upload_file(chandung, 'face_dataset/ChanDung')
    
    doc = {
      'msv': msv,
      'name': name,
      'TheSV': f"gs://demo2-2a1d9.appspot.com/face_dataset/TheSV/{thesv.name}",
      'ChanDung': f"gs://demo2-2a1d9.appspot.com/face_dataset/ChanDung/{chandung.name}",
      'feature': thesv_feature[0].tolist(),
      'feature_chandung': chandung_feature[0].tolist(),
    }
    
    self.db.insert(doc)
    return 1
  
  def update(self, id, msv: str, name: str, thesv: UploadedFile, chandung: UploadedFile):
    if thesv is not None:
      _thesv_img = Image.open(thesv)
      thesv_img = cv2.cvtColor(np.array(_thesv_img), cv2.COLOR_RGB2BGR)
      bboxs_thesv, thesv_img = self.detect(thesv_img)
      if len(bboxs_thesv) == 0:
        return -1
      if len(bboxs_thesv) > 1:
        return -2
      thesv_feature = self.regconize.infer(thesv_img, bboxs_thesv[0][:-1])
      thesv.seek(0)

    if chandung is not None:
      _chandung_img = Image.open(chandung)
      chandung_img = cv2.cvtColor(np.array(_chandung_img), cv2.COLOR_RGB2BGR)
      bboxs_chandung, chandung_img = self.detect(chandung_img)
      if len(bboxs_chandung) == 0:
        return -1
      if len(bboxs_chandung) > 1:
        return -2
      chandung_feature = self.regconize.infer(chandung_img, bboxs_chandung[0][:-1])
      chandung.seek(0)
    
    if thesv is not None:
      self.db.upload_file(thesv, 'face_dataset/TheSV')
      self.db.update(id, { \
                        'TheSV': f"gs://demo2-2a1d9.appspot.com/face_dataset/TheSV/{thesv.name}", \
                        'feature': thesv_feature[0].tolist(), \
                        })
      
    if chandung is not None:
      self.db.upload_file(chandung, 'face_dataset/ChanDung')
      self.db.update(id, { \
                        'ChanDung': f"gs://demo2-2a1d9.appspot.com/face_dataset/ChanDung/{chandung.name}",
                        'feature_chandung': chandung_feature[0].tolist(),
                        })
    
    if msv is not None:
      self.db.update(id, {'msv': msv})
    if name is not None:
      self.db.update(id, {'name': name})
      
    return 1
  
  def delete(self, ids):
    for id in ids:
      self.db.delete(id)
    return 1
  
  def parse_data(self):
    tb = {
      # "msv" : [],
      # "name" : [],
      # "TheSV" : [],
      # "ChanDung" : [],
      # "checkbox" : [],
      # "id": [],
      # "feature": [],
      # "feature_chandung": [],
    }

    for i in self.db.get_all():
      tb[i.id] = i.to_dict()
      # j = i.to_dict()
      # tb["feature_chandung"].append(j["feature_chandung"])
      # tb["checkbox"].append(False)
      # tb["msv"].append(j["msv"])
      # tb["name"].append(j["name"])
      # path1 = j["TheSV"].replace("gs://demo2-2a1d9.appspot.com/","")
      # path2 = j["ChanDung"].replace("gs://demo2-2a1d9.appspot.com/","")

      # public_url = self.db.bucket.blob(path1).generate_signed_url(expiration=timedelta(seconds=3300), method='GET')
      # tb["TheSV"].append(public_url)
      # tb["feature"].append(j["feature"])
      # public_url = self.db.bucket.blob(path2).generate_signed_url(expiration=timedelta(seconds=3600), method='GET')
      # tb["ChanDung"].append(public_url)

    return tb

  def find(self, msv: str, name: str):
    def remove_accents(input_str):
      nfkd_form = unicodedata.normalize('NFKD', input_str)
      return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    docs = self.db.get_all()
    results = {}
    
    msv = '.*' + re.sub(
                        r'\s+', '', remove_accents(
                                                    msv.replace('*', '.*').strip()
                                                  ).lower()
                        ) + '.*'
    
    name = '.*' + re.sub(
                          r'\s+', '', remove_accents(
                                                      name.replace('*', '.*').strip()
                                                    ).lower()
                        ) + '.*'
        
    for doc in docs:
      dat = doc.to_dict()
      dat_msv = re.sub(r'\s+', '', remove_accents(dat['msv']).lower())
      dat_name = re.sub(r'\s+', '', remove_accents(dat['name']).lower())

      m1 = (msv == "") or (re.match(msv, dat_msv) is not None)
      m2 = (name == "") or (re.match(name, dat_name) is not None)
      if m1 and m2:
        results[doc.id] = dat

    return results            
  
  def get_all(self):
    return self.db.get_all()