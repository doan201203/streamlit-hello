from __future__ import print_function
import streamlit as st
import cv2 as cv
import os
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.set_page_config(page_title="GrabCut")
st.markdown("# GrabCut")
st.sidebar.header("GrabCut")

#!/usr/bin/env python

'''
Welcome to the GrabCut demo!
'''
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

#doc anh tu nguoi dung
img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
if img is not None:
  #luu anh
  imgg = Image.open(img)
  ori_img = np.array(imgg)
  # tmp = ori_img.copy()
  # if len(ori_img.shape) == 3:
    # tmp = cv.cvtColor(ori_img, cv.COLOR_BGR2RGB)
  # imgg = Image.fromarray(tmp)
  # imgg.save('images/'+img.name)
  
  if imgg is not None:
    copy = ori_img.copy()
    mask2 = np.zeros(copy.shape[:2], dtype = np.uint8)
    #hien thi mode ve anh
    drawling_mode = st.sidebar.selectbox("Drawing mode", DRAWING_MODE)
    stroke_color = "red"
    
    #usage
    
    col1, col2 = st.columns(2)
    col1 = st.form(key='form', clear_on_submit=True)
    with col1 as form:
      canvas_rs = st_canvas(
        background_image=imgg,
        # update_streamlit=True,
        height=copy.shape[0],
        width=copy.shape[1], 
        display_toolbar=True,
        stroke_width=2,
        fill_color='',
        drawing_mode=CONVERSION[drawling_mode],
        stroke_color=stroke_color,
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
        if rec[i]['type'] == 'circle':
          fa = 1
          # print(rec[i])
          x = rec[i]['left']
          y = rec[i]['top']
          r = rec[i]['radius']
          ag = rec[i]['angle']
          cenx = x + r * np.cos(ag * np.pi / 180)
          ceny = y + r * np.sin(ag * np.pi / 180)
          print(cenx, ceny)
          color = rec[i]['fill']
          if color == 'black':
            co = cv.GC_BGD
          else:
            co = cv.GC_FGD
          cv.circle(mask2, (int(cenx), int(ceny)), r, co, -1)
        
      submit = st.form_submit_button('Submit')
      if submit:
        # print(max_one_rec, recc, fa, copy.shape)
        if max_one_rec > 0:
          if fa == 0:
            mask_type = cv.GC_INIT_WITH_RECT
          else:
            recc = (1, 1, copy.shape[0], copy.shape[1])
            mask_type = cv.GC_INIT_WITH_MASK
          bgdmodel = np.zeros((1, 65), np.float64)
          fgdmodel = np.zeros((1, 65), np.float64)
          # mask = np.zeros(copy.shape[:2], dtype = np.uint8)
          mask2[mask2 == 0] = cv.GC_BGD
          mask2[mask2 > 0] = cv.GC_PR_BGD
          
          cv.grabCut(copy, mask2, recc, bgdmodel, fgdmodel, 1, mask_type)
          alpha = np.where((mask2 == cv.GC_BGD) | (mask2==cv.GC_PR_BGD), 0, 255).astype('uint8')
          img_tmp = cv.bitwise_and(copy, copy, mask=alpha)
          # print(mask)
          # st.image(mask, caption="Edited")

          col2 = st.image(crop_to_alpha(alpha, img_tmp), caption="Edited")
        else:
          st.warning("Please draw a rectangle")