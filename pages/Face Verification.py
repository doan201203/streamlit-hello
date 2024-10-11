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

st.set_page_config(page_title="Face Verification", initial_sidebar_state="expanded", layout="wide")
st.title("Face Verification")

tools_key = [
  "display_add",
  "display_edit",
  "display_search",
  "display_delete",
  "display_home",
]

@st.cache_resource(show_spinner=False, ttl=3600)
def connect():
  db = firestore.Client.from_service_account_info(st.secrets)
  bucket = storage.Client.from_service_account_info(st.secrets).get_bucket('demo2-2a1d9.appspot.com')
  return db, bucket

db, bucket = connect()

def parse_data(doc: firestore.CollectionReference):
  tb = {
    "msv" : [],
    "name" : [],
    "TheSV" : [],
    "ChanDung" : [],
    "checkbox" : [],
    "id": []
  }

  for i in doc:
    tb["id"].append(i.id)
    j = i.to_dict()
    tb["checkbox"].append(False)
    tb["msv"].append(j["msv"])
    tb["name"].append(j["name"])
    path1 = j["TheSV"].replace("gs://demo2-2a1d9.appspot.com/","")
    path2 = j["ChanDung"].replace("gs://demo2-2a1d9.appspot.com/","")

    public_url = bucket.blob(path1).generate_signed_url(expiration=timedelta(seconds=3300), method='GET')
    tb["TheSV"].append(public_url)

    public_url = bucket.blob(path2).generate_signed_url(expiration=timedelta(seconds=3600), method='GET')
    tb["ChanDung"].append(public_url)

  return pd.DataFrame(tb)
  
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
    },
    use_container_width=True,
    disabled=("msv", "name", "TheSV", "ChanDung"),
    # key="table",
    hide_index=True,
  )
# sto_ref = 
st.header("1. CSDL")

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
    tools_bar[1].button("Làm mới", use_container_width=True, on_click=callb, args=("display_home",))
    tools_bar[2].button("Tìm kiếm", use_container_width=True, on_click=callb, args=("display_search",))
    tools_bar[3].button("Thêm", use_container_width=True, on_click=callb, args=("display_add",))
    tools_bar[4].button("Chỉnh sửa", use_container_width=True, on_click=callb, args=("display_edit",))
    tools_bar[5].button("Xóa", use_container_width=True, on_click=callb, args=("display_delete",))
  
  # 11 for tools form 12 for table  
  sec11 = st.container()
  sec12 = st.container()
  
  with sec12:
    if st.session_state.ctr != 0:
      st.session_state.ctr = 0
      st.session_state.df_value = get_all()
    tb = display_table(st.session_state.df_value)
  
  if st.session_state.display_add:
    def add():
      st.write("Thêm")
      with sec11:
        with st.form(key="add", clear_on_submit=True):
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
                
              bucket.blob(path1).upload_from_file(the_sv, content_type=the_sv.type)
              bucket.blob(path2).upload_from_file(chan_dung, content_type=chan_dung.type)
                
              db.collection('face_dataset').add({
                "msv": msv,
                "name": name,
                "TheSV": f"gs://demo2-2a1d9.appspot.com/{path1}",
                "ChanDung": f"gs://demo2-2a1d9.appspot.com/{path2}",
              })
              st.toast("Thêm thành công", icon=":material/check:")
              st.session_state.ctr = 1
              time.sleep(1.5)
              st.rerun()
            else:
              st.warning("Vui lòng cung cấp đầy đủ thông tin")
    add()
  elif st.session_state.display_edit:
    # Get checked rows
    st.session_state.df_value = tb
    checked = tb[tb["checkbox"]]
    
    if len(checked) == 0:
      st.warning("Chọn một dòng để chỉnh sửa")
      callb("display_home")
    elif len(checked) > 1:
      callb("display_home")
      st.warning("Chỉ được chọn một dòng để chỉnh sửa")
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
            col = st.form_submit_button("Add", use_container_width=True)
            id = db.collection('face_dataset').where(filter=firestore.FieldFilter("msv", "==", data.get("msv"))).where(filter=firestore.FieldFilter("name", "==", data.get("name"))).stream()
            ii = 0
            
            for dat in id:
              ii = dat.id
            
            if col:
                if the_sv:
                  path1 = f"face_dataset/TheSV/{the_sv.name}"
                  bucket.blob(path1).upload_from_file(the_sv, content_type=the_sv.type)
                  db.collection('face_dataset').document(ii).update({
                    "TheSV": f"gs://demo2-2a1d9.appspot.com/{path1}",
                  })
                if chan_dung:
                  path2 = f"face_dataset/ChanDung/{chan_dung.name}"
                  bucket.blob(path2).upload_from_file(chan_dung, content_type=chan_dung.type)
                  db.collection('face_dataset').document(ii).update({
                    "ChanDung": f"gs://demo2-2a1d9.appspot.com/{path2}",
                  })
                if msv:
                  db.collection('face_dataset').document(ii).update({
                    "msv": msv,
                  })
                if name:
                  db.collection('face_dataset').document(ii).update({
                    "name": name,
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
    def delete(checked: pd.DataFrame):  
      for i in checked:
        db.collection('face_dataset').document(i).delete()
      st.session_state.ctr = 1
      callb("display_home")
      st.toast("Xóa thành công", icon=":material/check:")
      time.sleep(1.5)
      st.rerun()
    
    checked = tb[tb["checkbox"] == True]["id"].tolist()
    if len(checked) == 0:
      st.warning("Vui lòng chọn ít nhất 1 dòng để xóa")
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
# display_table(st.session_state.df_value)

# st.write(get_all().to_html(escape=False, index=False), unsafe_allow_html=True)
