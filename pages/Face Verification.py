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
from my_utils.face_controller import FaceController

st.set_page_config(page_title="Face Verification", initial_sidebar_state="expanded", layout="wide")
st.title("Face Verification")

tools_key = [
  "display_add",
  "display_edit",
  "display_search",
  "display_delete",
  "home",
  "empty",
]


@st.cache_resource(show_spinner=False, ttl=3600)
def connect():
  controller = FaceController("face_dataset")
  # db = firestore.Client.from_service_account_info(st.secrets)
  # bucket = storage.Client.from_service_account_info(st.secrets).get_bucket('demo2-2a1d9.appspot.com')
  return controller

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

controller = connect()

def parse_data(doc: dict):
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

  for i, dat in doc.items():
    tb["id"].append(i)
    # j = i.to_dict()
    tb["feature_chandung"].append(dat["feature_chandung"])
    tb["checkbox"].append(False)
    tb["msv"].append(dat["msv"])
    tb["name"].append(dat["name"])
    path1 = dat["TheSV"].replace("gs://demo2-2a1d9.appspot.com/","")
    path2 = dat["ChanDung"].replace("gs://demo2-2a1d9.appspot.com/","")

    public_url = controller.db.bucket.blob(path1).generate_signed_url(expiration=timedelta(seconds=3300), method='GET')
    tb["TheSV"].append(public_url)
    tb["feature"].append(dat["feature"])
    public_url = controller.db.bucket.blob(path2).generate_signed_url(expiration=timedelta(seconds=3600), method='GET')
    tb["ChanDung"].append(public_url)

  return pd.DataFrame(tb)
  
def get_all():
  return parse_data(controller.parse_data())
  
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
    tools_bar[1].button("Làm mới", use_container_width=True, on_click=callb, args=("empty",), icon=":material/refresh:")
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
              with st.spinner("Đang xử lí"):
                sts = controller.insert(
                                msv=msv,
                                name=name,
                                thesv=the_sv,
                                chandung=chan_dung,
                                )
                if sts == -1:
                  st.toast("Không tìm thấy khuôn mặt, vui lòng thử ảnh khác")
                elif sts == -2:
                  st.toast("Tìm thấy nhiều khuôn mặt, vui lòng thử ảnh khác")
                else:
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
      callb("home")
    elif len(checked) > 1:
      callb("home")
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
            id = data.get("id")
            if col:
              with st.spinner("Đang xử lí"):
                sts = controller.update(id, msv if msv else None, name if name else None, the_sv, chan_dung)
                if sts == -1:
                  st.toast("Không tìm thấy khuôn mặt, vui lòng thử ảnh khác")
                elif sts == -2:
                  st.toast("Tìm thấy nhiều khuôn mặt, vui lòng thử ảnh khác")
                else:
                  st.session_state.ctr = 1
                  st.toast("Cập nhật thành công", icon=":material/check:")
                  callb("home")
                  time.sleep(1)
                  st.rerun()
            
      modify("Edit", checked.iloc[0])
      
  elif st.session_state.display_search:
    def search():
      with sec11:
        with st.form(key="search") as search_form:
          # st.write("Search")
          st.markdown("""
                      - Những thông tin không chắc chắn có thể thay thế bằng kí tự ' * '.
                      - Ví dụ: 21T*036, *036, 21T1* .
                      """)
          msv, name = st.columns(2)
          msv = msv.text_input("MSV")
          name = name.text_input("Name")
          sub = st.form_submit_button("Search", use_container_width=True)
          if sub:
            # like select * from face_dataset where msv like '%msv%' and name like '%name%'
            dt = controller.find(msv, name)
            print(dt)
            st.session_state.df_value = parse_data(dt)
            st.session_state.ctr = 0
            callb("home")
            time.sleep(1)
            st.rerun() 
            
    search()
  elif st.session_state.display_delete:
    @st.dialog("Bạn có chắc chắn muốn xóa không?")
    def delete(checked: pd.DataFrame):  
     
      if st.button("Xác nhận"):
        print(checked)
        controller.delete(checked)
        st.session_state.ctr = 1
        callb("home")
        st.toast("Xóa thành công", icon=":material/check:")
        time.sleep(1)
        st.rerun()
      else:
        callb("home")
    
    checked = tb[tb["checkbox"] == True]["id"].tolist()
    if len(checked) == 0:
      st.toast("Chọn ít nhất một dòng để xóa", icon=":material/warning:")
    else:
      callb("home")
      delete(checked)
      callb("home")
  elif st.session_state.home:
    for key in tools_key:
      if key.startswith("display"):
        st.session_state[key] = False
    st.session_state.home = False
    st.rerun()
  elif st.session_state.empty:
    st.session_state.empty = False
    st.cache_data.clear()
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
    # conf = st.slider("Confidence threshold cho phát hiện khuôn mặt", min_value=0.1, max_value=1.0, value=0.85, step=0.05)
    submit = st.form_submit_button("Xác thực", use_container_width=True)
    if submit:
      if cols[0] and cols[1]:
        with st.spinner('Đang xử lí'):
          detector = load_detector(0.85)
          regc = load_recognizer()
          
          card = Image.open(cols[0])
          chandung = Image.open(cols[1])
          card = cv.cvtColor(np.array(card), cv.COLOR_RGB2BGR)
          chandung = cv.cvtColor(np.array(chandung), cv.COLOR_RGB2BGR)
          # h, w = card.shape[:2]
          
          ver = Verification(detector, regc)
          ver.set_card(card)
          ver.set_selfie(chandung)
          card_face, fac, score, matches = ver.verify_card()
          
          # sf, car = ver.visualize()
          if card_face is not None:
            sf, car = ver.visualize(img1=ver.card_number, faces1=card_face, img2=ver.selfie, faces2=fac, matches=matches, scores=score)
            
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
          _img = img.copy()
          
          for dd in feature:
            features.append(regc.infer(img, bbox=dd[:-1]))
            face_box = dd[0:4].astype(np.int32)
            _img = cv.rectangle(_img,
                                (face_box[0], face_box[1]),
                                (face_box[0] + face_box[2], face_box[1] + face_box[3]),
                                (0, 0, 255),
                                2)
          features = np.asarray(features)

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
            for det in features:
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
              _img = cv.putText(_img, msv, (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
          st.write("Danh sách sinh viên có mặt trong lớp: ")
          st.write(pd.DataFrame(np.unique(p), columns=["msv"], index=np.arange(1, len(np.unique(p))+1)))
          
        st.image(_img, channels="BGR", use_column_width=True)

sec3()
  
