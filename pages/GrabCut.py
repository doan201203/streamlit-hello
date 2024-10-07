from __future__ import print_function
import streamlit as st
import cv2 as cv
import os
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
import numpy as np
from algorithm.grabcut import grabcut

st.set_page_config(page_title="GrabCut")
st.markdown("# GrabCut")
st.sidebar.header("GrabCut")

#!/usr/bin/env python

# st.image()
def crop_to_alpha(alpha, img):
  x, y = alpha.nonzero()
  if len(x) == 0 or len(y) == 0: return img
  return img[np.min(x) : np.max(x), np.min(y) : np.max(y)]

BLUE  = [255, 0, 0]       # rectangle color
RED   = [0, 0, 255]       # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG    = {'color' : BLACK, 'val' : 0}
DRAW_FG    = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED,   'val' : 2}
DRAWING_MODE = ['Draw a rectangle']
CONVERSION = {
  'Draw a rectangle': 'rect',
  # 'Draw touchup curves': 'point'
}


thickness  = 3

st.write("Huớng dẫn sử dụng")
st.write('1. Upload ảnh cần xử lí')
st.write('2. Vẽ 1 hình chữ nhật xung quanh đối tượng cần tách khỏi nền')
st.write('3. Ở bên dưới ảnh có các nút ↶, ↷, 🗑 tương ứng với các chức năng Undo, Redo, Reset hình chữ nhật đã vẽ')
st.write('4. Sau khi vẽ xong hình chữ nhật bấm nút Submit để áp dụng thuật toán')

#doc anh tu nguoi dung
img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
if img is not None:
  #luu anh
  imgg = Image.open(img)
  ori_img = np.array(imgg)

  copy = ori_img.copy()
  mask2 = np.zeros(copy.shape[:2], dtype = np.uint8)
  #hien thi mode ve anh
  drawling_mode = st.sidebar.selectbox("Drawing mode", DRAWING_MODE)
  stroke_color = "red"
  
  #usage
  
  col1, col2 = st.columns(2)
  col1 = st.form(key='form', clear_on_submit=True)
  #scale if lager image
  h, w = 550, 700
  reh, rew = ori_img.shape[0], ori_img.shape[1]
  if reh > rew:
    h, w = 700, 550
  ori_img = cv.resize(ori_img, (w, h))
  
  with col1 as form:
    canvas_rs = st_canvas(
      background_image=imgg,
      display_toolbar=True,
      stroke_width=2,
      fill_color='',
      drawing_mode=CONVERSION[drawling_mode],
      stroke_color=stroke_color,
      height=h,
      width=w,
      # key="my_canvas",
    )
  
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
      # if rec[i]['type'] == 'circle':
      #   fa = 1
      #   # print(rec[i])
      #   x = rec[i]['left']
      #   y = rec[i]['top']
      #   r = rec[i]['radius']
      #   ag = rec[i]['angle']
      #   cenx = x + r * np.cos(ag * np.pi / 180)
      #   ceny = y + r * np.sin(ag * np.pi / 180)
      #   print(cenx, ceny)
      #   color = rec[i]['fill']
      #   if color == 'black':
      #     co = cv.GC_BGD
      #   else:
      #     co = cv.GC_FGD
        # cv.circle(mask2, (int(cenx), int(ceny)), r, co, -1)
      
    submit = st.form_submit_button('Submit')
    if submit:
      if max_one_rec > 0:
        mask_type = cv.GC_INIT_WITH_RECT
        
        # ori_img = cv.resize(ori_img, (h, h))
        #size less than then dont resize
        
        res = grabcut(
          ori_img, rect=recc
        )
        # print(recc)
        # cv.rectangle(ori_img, (recc[0], recc[1]), (recc[0] + recc[2], recc[1] + recc[3]), (0, 255, 0), 2)
        
        col2 = st.image(res, caption="Edited")
      else:
        st.warning("Please draw a rectangle")
