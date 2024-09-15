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

  # alpha = np.where((mask == 1) + (mask == 3), 255,
                                  # 0).astype('uint8')
  # im = cv.bitwise_and(copy, copy, mask=alpha)
  # return crop_to_alpha(alpha, copy)
  mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
  copy = copy*mask2[:,:,np.newaxis]
  return copy

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
  'Draw touchup curves': 'point'
}

thickness  = 3

def grcut():
  #doc anh tu nguoi dung
  img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
  if img is not None:
    #luu anh
    if not os.path.exists('images'):
      os.makedirs('images')
    imgg = Image.open(img)
    imgg.save('images/'+img.name)
    ori_img = cv.imread('images/'+img.name)
    
    if imgg is not None:
      copy = np.asarray(imgg)
      mask = np.zeros(copy.shape[:2], dtype = np.uint8)
      #hien thi mode ve anh
      drawling_mode = st.sidebar.selectbox("Drawing mode", DRAWING_MODE)
      stroke_color = "red"
      if drawling_mode == DRAWING_MODE[1]:
        drawling_mode2 = st.sidebar.selectbox("Drawing mode", ('Select areas of sure background', 'Select areas of sure foreground'))
        if drawling_mode2 == 'Select areas of sure background':
          stroke_color = "black"
        else:
          stroke_color = "green"
      canvas_rs = st_canvas(
        background_image=Image.open(img),
        update_streamlit=True,
        height=imgg.height,
        width=imgg.width, 
        display_toolbar=True,
        fill_color='' if drawling_mode == DRAWING_MODE[0] else 'black' if drawling_mode2 == 'Select areas of sure background' else 'green',
        stroke_width=2,
        drawing_mode=CONVERSION[drawling_mode],
        stroke_color=stroke_color,
        key="my_canvas"
      )
      
      #
      #
      form = st.form(key='form')
      # print(canvas_rs)
      rec = []
      if canvas_rs.json_data is not None:
        rec = canvas_rs.json_data['objects']
      
      recc = ()
      for i in rec:
        if i['type'] == 'rect':
          x = i['left']
          y = i['top']
          w = i['width']
          h = i['height']
          recc = (min(x, x + w), min(y, y + h), w, h)
          # cv.rectangle(cmp, (x, y), (x+w, y+h), (255, 255, 255), 2)
      
      max_one_rec = 0
      fa = 0
      
      for i in range(len(rec)):
        if rec[i]['type'] == 'rect':
          max_one_rec += 1
          if max_one_rec > 1:
            st.warning("Only one rectangle is allowed")
            rec.pop(i)
        if rec[i]['type'] == 'circle':
          fa = 1
          x = rec[i]['left']
          y = rec[i]['top']
          r = rec[i]['radius']
          color = rec[i]['fill']
          if color == 'black':
            color = 0
          else:
            color = 1
          cv.circle(mask, (x, y), r, color, -1)
      
      submit = form.form_submit_button('Submit')
      if submit:
        if max_one_rec > 0:
          if fa == 0:
            mask_type = cv.GC_INIT_WITH_RECT
          else:
            mask_type = cv.GC_INIT_WITH_MASK
          bgdmodel = np.zeros((1, 65), np.float64)
          fgdmodel = np.zeros((1, 65), np.float64)
          # mask = np.zeros(copy.shape[:2], dtype = np.uint8)
          cv.grabCut(copy, mask, recc, bgdmodel, fgdmodel, 1, mask_type)
          alpha = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
          img_tmp = cv.bitwise_and(copy, copy, mask=alpha)
          # print(fa, np.where(mask.nonzero()))
          # st.image(cmp, caption="Edited")
          st.image(crop_to_alpha(alpha, img_tmp), caption="Edited")
        else:
          st.warning("Please draw a rectangle")
        
        # st.image(algo(copy, recc), caption="Edited")
      
      # st.write(img.getvalue())
      # App().run(img.name)
      # cv.destroyAllWindows()
grcut()