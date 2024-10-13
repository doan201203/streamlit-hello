import streamlit as st
from google.cloud import firestore, storage
import json
import toml
import pandas as pd
from datetime import timedelta
import requests
from PIL import Image
from google.cloud.firestore import FieldFilter as fil
import time
import threading
import cv2 as cv
from miscs.face_models.yunet import YuNet
from miscs.face_models.sface import SFace
from my_utils.card_verify import Verification
import numpy as np
import functions_framework
from cloudevents.http.event import CloudEvent
import firebase_admin as fa
from firebase_admin import credentials

from firebase_functions.firestore_fn import (
  on_document_created,
  on_document_deleted,
  on_document_updated,
  on_document_written, 
  Event,
  Change,
  DocumentSnapshot,
)

st.set_page_config(page_title="Face Verification", initial_sidebar_state="expanded", layout="wide")
st.title("Face Verification")

tools_key = [
  "display_add",
  "display_edit",
  "display_search",
  "display_delete",
  "display_home",
]

# app = fa.initialize_app()

@st.cache_resource(show_spinner=False, ttl=3600)
def connect():
  db = firestore.Client.from_service_account_info(st.secrets)
  bucket = storage.Client.from_service_account_info(st.secrets).get_bucket('demo2-2a1d9.appspot.com')
  return db, bucket

MODELS_PATH = "./miscs/face_models"
BACKEND_TARGET_PAIR = [
  [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
  [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
  [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
  [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
  [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU],
]

@st.cache_resource(show_spinner=False)
def load_detector(conf):
  backend_id, target_id = BACKEND_TARGET_PAIR[0]
  model = YuNet(modelPath=f"{MODELS_PATH}/face_detection_yunet_2023mar.onnx", 
                confThreshold=conf,
                topK=5000,
                backendId=backend_id,
                targetId=target_id,
                )
  return model
@st.cache_resource(show_spinner=False)
def load_recognizer():
  backend_id, target_id = BACKEND_TARGET_PAIR[0]
  model = SFace(backendId=backend_id,
                targetId=target_id,
                modelPath=f"{MODELS_PATH}/face_recognition_sface_2021dec.onnx",
                disType=0,
                )
  return model

db, bucket = connect()

def parse_data(doc: firestore.CollectionReference):
  tb = {
    "msv" : [],
    "name" : [],
    "TheSV" : [],
    "ChanDung" : [],
    "checkbox" : [],
    "id": [],
    "feature": [],
    "feature_chandung": [],
  }

  for i in doc:
    tb["id"].append(i.id)
    j = i.to_dict()
    tb["feature_chandung"].append(j["feature_chandung"])
    tb["checkbox"].append(False)
    tb["msv"].append(j["msv"])
    tb["name"].append(j["name"])
    path1 = j["TheSV"].replace("gs://demo2-2a1d9.appspot.com/","")
    path2 = j["ChanDung"].replace("gs://demo2-2a1d9.appspot.com/","")

    public_url = bucket.blob(path1).generate_signed_url(expiration=timedelta(seconds=3300), method='GET')
    tb["TheSV"].append(public_url)
    tb["feature"].append(j["feature"])
    public_url = bucket.blob(path2).generate_signed_url(expiration=timedelta(seconds=3600), method='GET')
    tb["ChanDung"].append(public_url)

  return pd.DataFrame(tb)
  
@on_document_updated(document="face_dataset/{docId}/")
def store_embedding(event : Event[Change[DocumentSnapshot]])->None:
  a = "new"
  print(a)
  print("moi", event.data.after.to_dict())

@on_document_created(document="face_dataset/{docId}")
def store_embedding(event : Event[DocumentSnapshot])->None:
  a = "new"
  print(a)
  print("moi", event.data.after.to_dict())

def get_all():
  return parse_data(db.collection('face_dataset').stream())
  
def display_table(tb):
  return st.data_editor(
    tb,
    column_config= {
      "checkbox": st.column_config.CheckboxColumn("Chon"),
      "msv": st.column_config.TextColumn("MSV"),
      "name": st.column_config.TextColumn("Name"),
      "TheSV": st.column_config.ImageColumn("TheSV"),
      "ChanDung": st.column_config.ImageColumn("ChanDung"),
      "id": None,
      "feature": None,
      "feature_chandung": None,
    },
    use_container_width=True,
    disabled=("msv", "name", "TheSV", "ChanDung"),
    # key="table",
    hide_index=True,
  )
# sto_ref = 
def get_feature(img):
  det = load_detector(0.7)
  det.setInputSize((img.shape[1], img.shape[0]))
  feature = det.infer(img)
  return feature

st.header("1. CSDL Sinh Viên", divider="green")

if "df_value" not in st.session_state:
    st.session_state.df_value = get_all()
    st.session_state.check_len = 0
    st.session_state.ctr = 0
    st.session_state.prev_op = None

for key in tools_key:
  if key not in st.session_state:
    st.session_state[key] = False

def sec1():
  # Check if any key is True
  def callb(key):
    for k in tools_key:
      if k != key and st.session_state[k]:
        st.session_state[k] = False
    st.session_state[key] = True
    st.session_state.prev_op = key
  
  with st.container():
    tools_bar = st.columns([5, 1, 1, 1, 1, 1], gap='small')
    tools_bar[1].button("Làm mới", use_container_width=True, on_click=callb, args=("display_home",), icon=":material/refresh:")
    tools_bar[2].button("Tìm kiếm", use_container_width=True, on_click=callb, args=("display_search",), icon=":material/search:")
    tools_bar[3].button("Thêm", use_container_width=True, on_click=callb, args=("display_add",), icon=":material/add:")
    tools_bar[4].button("Chỉnh sửa", use_container_width=True, on_click=callb, args=("display_edit",), icon=":material/edit:")
    tools_bar[5].button("Xóa", use_container_width=True, on_click=callb, args=("display_delete",), icon=":material/delete:")
  
  # 11 for tools form 12 for table  
  sec11 = st.container()
  sec12 = st.container()
  
  with sec12:
    if st.session_state.ctr != 0:
      st.session_state.ctr = 0
      st.session_state.df_value = get_all()
    print(st.session_state.df_value)
    if len(st.session_state.df_value) == 0:
      st.write("Không có dữ liệu")
    else:
      tb = display_table(st.session_state.df_value)
  
  if st.session_state.display_add:
    def add():
      st.write("Thêm")
      with sec11:
        with st.form(key="add", clear_on_submit=False):
          cols = st.columns(2)
          msv = cols[0].text_input("MSV")
          name = cols[1].text_input("Name")
          
          the_sv = cols[0].file_uploader("TheSV")
          chan_dung = cols[1].file_uploader("ChanDung")
          col = st.columns([8, 1, 1])
          col[1] = st.form_submit_button("Thêm", use_container_width=True)
          
          if col[1]:
            print("HERE")
            if the_sv and chan_dung and msv and name:
              path1 = f"face_dataset/TheSV/{the_sv.name}"
              path2 = f"face_dataset/ChanDung/{chan_dung.name}"
                # print(path1, path2, the_sv.t)
              img1 = Image.open(the_sv)
              img1 = cv.cvtColor(np.array(img1), cv.COLOR_RGB2BGR)
              max_width = 640
              scale = max_width / img1.shape[1]
              img1 = cv.resize(img1, (max_width, int(img1.shape[0] * scale)))
              feature = get_feature(img1)

              img2 = Image.open(chan_dung)
              img2 = cv.cvtColor(np.array(img2), cv.COLOR_RGB2BGR)
              scale = max_width / img2.shape[1]
              img2 = cv.resize(img2, (max_width, int(img2.shape[0] * scale)))
              feature2 = get_feature(img2)
              # print(feature)
              if len(feature) == 0 or len(feature2) == 0:
                st.toast("Không tìm thấy khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
                time.sleep(1)
              elif len(feature) > 1 or len(feature2) > 1:
                st.toast("Tìm thấy nhiều khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
                time.sleep(1)
              else:
                the_sv.seek(0)
                chan_dung.seek(0)
                # print(feature[0][-1])
                bucket.blob(path1).upload_from_file(the_sv, content_type=the_sv.type)
                bucket.blob(path2).upload_from_file(chan_dung, content_type=chan_dung.type)
                regc = load_recognizer()
                feature = regc.infer(img1, bbox=feature[0][:-1])
                feature2 = regc.infer(img2, bbox=feature2[0][:-1])
                db.collection('face_dataset').add({
                  "msv": msv,
                  "name": name,
                  "TheSV": f"gs://demo2-2a1d9.appspot.com/{path1}",
                  "ChanDung": f"gs://demo2-2a1d9.appspot.com/{path2}",
                  "feature": feature[0].tolist(),
                  "feature_chandung": feature2[0].tolist(),
                })
                st.toast("Thêm thành công", icon=":material/check:")
                st.session_state.ctr = 1
                time.sleep(1.5)
                st.rerun()
            else:
              st.toast("Vui lòng cung cấp đầy đủ thông tin", icon=":material/warning:")
    add()
  elif st.session_state.display_edit:
    # Get checked rows
    st.session_state.df_value = tb
    checked = tb[tb["checkbox"]]
    
    if len(checked) == 0:
      st.toast("Chọn một dòng để chỉnh sửa", icon=":material/warning:")
      callb("display_home")
    elif len(checked) > 1:
      callb("display_home")
      st.toast.warning("Chỉ được chọn tối đa một dòng để chỉnh sửa", icon=":material/warning:")
    else:

      def modify(title, data: pd.Series):
        with sec11:
          st.write(title)
          with st.form(key="modify") as modify_form:
            cols = st.columns(2)
            msv = cols[0].text_input("MSV", data.get("msv"))
            name = cols[1].text_input("Name", data.get("name"))

            img = Image.open(requests.get(data.get("TheSV"), stream=True).raw)
            img2 = Image.open(requests.get(data.get("ChanDung"), stream=True).raw)
            #resize
            img.thumbnail((200, 200))
            img2.thumbnail((200, 200))
            
            img_col = st.columns(2)
            if img:
              img_col[0].image(img, caption="TheSV")
            if img2:
              img_col[1].image(img2, caption="ChanDung")
            
            cols = st.columns(2)     
            the_sv = cols[0].file_uploader("TheSV", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Upload an image")
            chan_dung = cols[1].file_uploader("ChanDung", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Upload an image")
            col = st.form_submit_button("Xác nhận", use_container_width=True)
            id = db.collection('face_dataset').where(filter=firestore.FieldFilter("msv", "==", data.get("msv"))).where(filter=firestore.FieldFilter("name", "==", data.get("name"))).stream()
            ii = 0
            
            for dat in id:
              ii = dat.id
            
            if col:
              feature, feature2 = None, None
              ok1, ok2 = False, False
              if the_sv:
                img1 = Image.open(the_sv)
                img1 = cv.cvtColor(np.array(img1), cv.COLOR_RGB2BGR)
                feature = get_feature(img1)
                
                if len(feature) == 0:
                  st.toast("Không tìm thấy khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
                  time.sleep(1)
                  st.rerun()
                elif len(feature) > 1:
                  #toi da 1 khuon mat
                  st.toast("Tìm thấy nhiều khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
                  time.sleep(1)
                  st.rerun()
                else:
                  ok1 = True
              if chan_dung:
                img2 = Image.open(chan_dung)
                img2 = cv.cvtColor(np.array(img2), cv.COLOR_RGB2BGR)
                max_width = 640
                scale = max_width / img2.shape[1]
                img2 = cv.resize(img2, (max_width, int(img2.shape[0] * scale)))
                feature2 = get_feature(img2)
                if len(feature2) == 0:
                  st.toast("Không tìm thấy khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
                  time.sleep(1)
                  st.rerun()
                elif len(feature2) > 1:
                  st.toast("Tìm thấy nhiều khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
                  time.sleep(1)
                  st.rerun()
                else:
                  ok2 = True
              
              if (not ok1 and feature is not None) or (not ok2 and feature2 is not None):
                st.toast("Lỗi xác thực khuôn mặt, vui lòng chọn ảnh khác", icon=":material/warning:")
              else:
                
                if msv:
                  db.collection('face_dataset').document(ii).update({
                    "msv": msv,
                  })
                if name:
                  db.collection('face_dataset').document(ii).update({
                    "name": name,
                  })
                if feature is not None:
                  regc = load_recognizer()
                  feature = regc.infer(img1, bbox=feature[0][:-1])
                  the_sv.seek(0)
                  path1 = f"face_dataset/TheSV/{the_sv.name}"
                  bucket.blob(path1).upload_from_file(the_sv, content_type=the_sv.type)
                  db.collection('face_dataset').document(ii).update({
                    "TheSV": f"gs://demo2-2a1d9.appspot.com/{path1}",
                  })
                  db.collection('face_dataset').document(ii).update({
                    "feature": feature[0].tolist(),
                  })
                if feature2 is not None:
                  regc = load_recognizer()
                  feature2 = regc.infer(img2, bbox=feature2[0][:-1])
                  chan_dung.seek(0)
                  path2 = f"face_dataset/ChanDung/{chan_dung.name}"
                  bucket.blob(path2).upload_from_file(chan_dung, content_type=chan_dung.type)
                  db.collection('face_dataset').document(ii).update({
                    "ChanDung": f"gs://demo2-2a1d9.appspot.com/{path2}",
                  })
                  db.collection('face_dataset').document(ii).update({
                    "feature_chandung": feature2[0].tolist(),
                  })
                  
                st.session_state.ctr = 1
                st.toast("Cập nhật thành công", icon=":material/check:")
                callb("display_home")
                time.sleep(1.5)
                st.rerun()
                
      modify("Edit", checked.iloc[0])
  elif st.session_state.display_search:
    def search():
      with sec11:
        with st.form(key="search") as search_form:
          st.write("Search")
          msv, name = st.columns(2)
          msv = msv.text_input("MSV")
          name = name.text_input("Name")
          sub = st.form_submit_button("Search", use_container_width=True)
          if sub:
            # like select * from face_dataset where msv like '%msv%' and name like '%name%'
            maxx = msv + '\uf8ff'
            maxx2 = name + '\uf8ff'
            dt = db.collection('face_dataset') \
                                              .where(filter=fil("msv", ">=", msv)) \
                                              .where(filter=fil("msv", "<=", maxx)) \
                                              .where(filter=fil("name", ">=", name)) \
                                              .where(filter=fil("name", "<=", maxx2)) \
                                              .stream()
            st.session_state.df_value = parse_data(dt)
            st.session_state.ctr = 0
            callb("display_home")
            time.sleep(1.5)
            st.rerun() 
            
    search()
  elif st.session_state.display_delete:
    @st.dialog("Bạn có chắc chắn muốn xóa không?")
    def delete(checked: pd.DataFrame):  
     
      if st.button("Xác nhận"):
        for i in checked:
          db.collection('face_dataset').document(i).delete()
        st.session_state.ctr = 1
        callb("display_home")
        st.toast("Xóa thành công", icon=":material/check:")
        time.sleep(1.5)
        st.rerun()
      else:
        callb("display_home")
    
    checked = tb[tb["checkbox"] == True]["id"].tolist()
    if len(checked) == 0:
      st.toast("Chọn ít nhất một dòng để xóa", icon=":material/warning:")
    else:
      callb("display_home")
      delete(checked)
      callb("display_home")
  elif st.session_state.display_home:
    callb("empty")
    st.cache_data.clear()
    # Reset
    st.session_state.ctr = 1
    st.rerun()

sec1()    


st.header("2. Xác thực thẻ sinh viên và ảnh chân dung ", divider="red")
def sec2():
  # callback_done = threading.Event()
  # def on_snapshot(doc_snapshot, changes, read_time):
  #   # print("NEW", doc_snapshot)
  #   for doc in doc_snapshot:
  #     print(f'Received document snapshot: {doc.id}')
  #   callback_done.set()
  # doc_watch = db.collection('face_dataset').on_snapshot(on_snapshot)
  
  with st.form(key='card_verify'):
    cols = st.columns(2)
    cols[0] = cols[0].file_uploader("Ảnh thẻ sinh viên", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Upload an image")
    cols[1] = cols[1].file_uploader("Ảnh chân dung", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Upload an image")
    conf = st.slider("Confidence threshold cho phát hiện khuôn mặt", min_value=0.1, max_value=1.0, value=0.85, step=0.05)
    submit = st.form_submit_button("Xác thực", use_container_width=True)
    if submit:
      if cols[0] and cols[1]:
        detector = load_detector(conf)
        regc = load_recognizer()
        
        card = Image.open(cols[0])
        chandung = Image.open(cols[1])
        card = cv.cvtColor(np.array(card), cv.COLOR_RGB2BGR)
        chandung = cv.cvtColor(np.array(chandung), cv.COLOR_RGB2BGR)
        # h, w = card.shape[:2]
        max_width = 640
        scale = max_width / chandung.shape[1]
        chandung = cv.resize(chandung, (max_width, int(chandung.shape[0] * scale)))
        
        ver = Verification(detector, regc)
        ver.set_card(card)
        ver.set_selfie(chandung)
        card_face, fac, score, matches = ver.verify_card()
        
        # sf, car = ver.visualize()
        if card_face is not None:
          sf, car = ver.visualize(img1=card, faces1=card_face, img2=chandung, faces2=fac, matches=matches, scores=score)
          
          col2 = st.columns(2)
          col2[1].image(sf, caption="Ảnh chân dung", channels="BGR")
          col2[0].image(car, caption="Ảnh thẻ sinh viên", channels="BGR")
        else:
          st.toast("Không tìm thấy khuôn mặt", icon=":material/warning:")
          time.sleep(1)
      else:
        st.toast("Vui lòng cung cấp đủ ảnh", icon=":material/warning:")
        time.sleep(1)
    
sec2()

st.header("3. Xác thực khuôn mặt trong lớp học", divider="blue")
def sec3():
  with st.form('aa'):
  # print(features[0], y[0])
    file = st.file_uploader("Ảnh cần xác thực", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Upload an image")
    if st.form_submit_button("Xác thực", use_container_width=True):
      p = []
      features = []
      det = load_detector(0.7)
      y = []
      regc = load_recognizer()
      for sv in st.session_state.df_value.iterrows():
        features.append(sv[1]["feature"])
        y.append(sv[1]["msv"])    
      features = np.asarray(features)
      y = np.asarray(y)
      
      if file is not None:
        with st.spinner("Đang xác thực..."):
          img = Image.open(file)
          img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
          
          det.setInputSize((img.shape[1], img.shape[0]))
          feature = det.infer(img)
          features = []
          for dd in feature:
            features.append(regc.infer(img, bbox=dd[:-1]))
          features = np.asarray(features)
          _img = img.copy()
          # print("Im", feature)
          for sv in st.session_state.df_value.iterrows():
            sv = sv[1]
            i = 0
            max_score, bbox, msv = 0, None, None
            for det in features:
              mark, lb = regc.match_f(np.array([sv["feature"]], dtype=np.float32), det)
              print(mark, lb)
              if lb == 1:
                if mark > max_score:
                  max_score = mark
                  bbox = feature[i][0:4].astype(np.int32)
                  msv = sv["msv"]
              i += 1
            i = 0
            max_score, bbox, msv = 0, None, None
            for det in features:
              # print("DETT", det, sv["feature_chandung"])
              mark, lb = regc.match_f(np.array([sv["feature_chandung"]], dtype=np.float32), det)
              print(mark, lb)
              if lb == 1:
                if mark > max_score:
                  max_score = mark
                  bbox = feature[i][0:4].astype(np.int32)
                  msv = sv["msv"]
              i += 1
            if bbox is not None:
              p.append(sv['msv'])
              _img = cv.rectangle(_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
              _img = cv.putText(_img, msv, (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0 , 255), 2)
          st.write("Danh sách sinh viên có mặt trong lớp: ")
          st.write(pd.DataFrame(np.unique(p), columns=["msv"], index=np.arange(1, len(np.unique(p))+1)))
          # for i in np.unique(p):
            # s(i)
        st.image(_img, channels="BGR", use_column_width=True)

sec3()
  
