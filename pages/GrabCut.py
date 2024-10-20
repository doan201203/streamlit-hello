from __future__ import print_function
import streamlit as st
import cv2 as cv
import os
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
import numpy as np
from algorithm.grabcut import Grabcut

st.set_page_config(page_title="GrabCut", layout="wide")
st.markdown("# GrabCut")
st.sidebar.header("GrabCut")

#!/usr/bin/env python

# st.image()

BLUE  = [255, 0, 0]       # rectangle color
RED   = [0, 0, 255]       # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG    = {'color' : BLACK, 'val' : 0}
DRAW_FG    = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED,   'val' : 2}
DRAWING_MODE = ['Draw a rectangle', 'Draw touchup curves']
CONVERSION = {
  'Draw a rectangle': 'rect',
  'Draw touchup curves': 'freedraw'
}

@st.cache_data(show_spinner=False)
def load_Grabcut(img):
  return Grabcut(img)

thickness  = 3

st.write("Huớng dẫn sử dụng")
st.write('1. Upload ảnh cần xử lí')
st.write('2. Vẽ 1 hình chữ nhật xung quanh đối tượng cần tách khỏi nền')
st.write('3. Ở bên dưới ảnh có các nút ↶, ↷, 🗑 tương ứng với các chức năng Undo, Redo, Reset hình chữ nhật đã vẽ')
st.write('4. Sau khi vẽ xong hình chữ nhật bấm nút Submit để áp dụng thuật toán')
st.write('5. Để chỉnh sửa các vùng Foreground hay Background chọn chế độ vẽ Touchup curves ở thanh slidebar bên trái màn hình và Submit lại để cập nhật')

img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

if img is not None:
  imgg = Image.open(img)
  ori_img = np.array(imgg)

  # Drawing mode  
  drawling_mode = st.sidebar.selectbox("Drawing mode", DRAWING_MODE)
  stroke_color = "red"
  
  if drawling_mode == 'Draw touchup curves':
    touchup = st.sidebar.radio("Touchup curves", ('Foreground', 'Background'))
    if touchup == 'Foreground':
      stroke_color = "green"
    else:
      stroke_color = "black"

  #scale if lager image
  w = min(550, ori_img.shape[1]) 
  h = w * ori_img.shape[0] // ori_img.shape[1] 
  ori_img = cv.resize(ori_img, (w, h))
  imgg.resize((w, h))
  
  extractor = load_Grabcut(ori_img)
  if "img_name" not in st.session_state:
    st.session_state['img_name'] = img
    st.session_state['extractor'] = extractor
    st.session_state['prev_img'] = None
    st.session_state['prev_name'] = img.name
    if 'my_canvas' in st.session_state:
        del st.session_state['my_canvas']
  else:
    if st.session_state['img_name'] != img:
      st.session_state['img_name'] = img
      st.session_state['extractor'] = extractor
      if st.session_state['prev_name'] in st.session_state:
        del st.session_state[st.session_state['prev_name']]
      st.session_state['prev_img'] = None
      st.session_state['prev_name'] = img.name
  
  with st.form(key="form"):
    col = st.columns(2, gap='large')
    with col[0]:
      canvas_rs = st_canvas(
        background_image=imgg,
        display_toolbar=True,
        stroke_width=6,
        fill_color='' if drawling_mode== DRAWING_MODE[0] else stroke_color,
        drawing_mode=CONVERSION[drawling_mode],
        stroke_color=stroke_color,
        height=h,
        width=w,
        update_streamlit=True,
        key=img.name,
      )

      # Get all data from canvas
      rec = []
      if canvas_rs.json_data is not None:
        rec = canvas_rs.json_data['objects']
      
      recc = ()
      fa = 0
      max_one_rec = 0
      for i in range(len(rec)):
        if rec[i]['type'] == 'rect':
          max_one_rec += 1
          x = rec[i]['left']
          y = rec[i]['top']
          w = rec[i]['width']
          h = rec[i]['height']
          recc = (min(x, x + w), min(y, y + h), w, h)
        if rec[i]['type'] == 'path':
          fa = 1
          color = rec[i]['stroke']
          path = rec[i]['path']
          if color == 'black':
            co = cv.GC_BGD
          else:
            co = cv.GC_FGD
          st.session_state['extractor'].path(path, co)
      
      submit = st.form_submit_button('Submit')
      
      
      # Display previous image edited
      if st.session_state['prev_img'] is not None:
        with col[1]:
          col[1] = st.image(st.session_state['prev_img'], caption="Edited")  
      
      if submit:
        if max_one_rec > 0:
          mask_type = cv.GC_INIT_WITH_RECT if fa == 0 else cv.GC_INIT_WITH_MASK
          
          st.session_state['extractor'].set_rect(recc)
          with st.spinner("Processing..."):
            res = st.session_state['extractor'].grabcut(
              type=0 if mask_type == cv.GC_INIT_WITH_RECT else cv.GC_INIT_WITH_MASK
            )
          
          st.session_state['prev_img'] = res
          with col[1]:
            col[1] = st.image(res, caption="Edited")
        else:
          st.warning("Please draw a rectangle")
