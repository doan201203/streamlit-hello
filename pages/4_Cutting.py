from __future__ import print_function
import streamlit as st
import cv2 as cv
import os
from PIL import Image 
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Cutting")
st.markdown("# Cutting")
st.sidebar.header("Cutting")

#!/usr/bin/env python
'''
===============================================================================
# grabcut

A simple program for interactively removing the background from an image using
the grab cut algorithm and OpenCV.

This code was derived from the Grab Cut example from the OpenCV project.

## Usage
    grabcut.py <input> [output]

## Operation

At startup, two windows will appear, one for input and one for output.

To start, in input window, draw a rectangle around the object using mouse right
button.  For finer touch-ups, press any of the keys below and draw circles on
the areas you want.  Finally, press 's' to save the result.

## Keys
  * 0 - Select areas of sure background
  * 1 - Select areas of sure foreground
  * 2 - Select areas of probable background
  * 3 - Select areas of probable foreground
  * n - Update the segmentation
  * r - Reset the setup
  * s - Save the result
  * q - Quit
===============================================================================
'''

# Python 2/3 compatibility


import numpy as np
import cv2 as cv
import sys


class App():
    BLUE  = [255, 0, 0]       # rectangle color
    RED   = [0, 0, 255]       # PR BG
    GREEN = [0, 255, 0]       # PR FG
    BLACK = [0, 0, 0]         # sure BG
    WHITE = [255, 255, 255]   # sure FG

    DRAW_BG    = {'color' : BLACK, 'val' : 0}
    DRAW_FG    = {'color' : WHITE, 'val' : 1}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
    DRAW_PR_BG = {'color' : RED,   'val' : 2}

    thickness  = 3


    def onmouse(self, event, x, y, flags, param):
        # Draw rectangle
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.input = self.copy.copy()
                cv.rectangle(self.input, (self.ix, self.iy), (x, y), self.BLUE,
                             2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x),
                             abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.input, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x),
                         abs(self.iy - y))
            self.rect_or_mask = 0
            self.segment()

        # Draw touchup curves
        if event == cv.EVENT_LBUTTONDOWN:
            if not self.rect_over: print('First draw a rectangle')

            else:
                self.drawing = True
                cv.circle(self.input, (x,y), self.thickness,
                          self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness,
                          self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.input, (x, y), self.thickness,
                          self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness,
                          self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.input, (x, y), self.thickness,
                          self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness,
                          self.value['val'], -1)
                self.segment()


    def reset(self):
        print('Resetting')

        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rectangle = False
        self.rect_or_mask = 100
        self.rect_over = False
        self.value = self.DRAW_FG

        self.input = self.copy.copy()
        self.mask = np.zeros(self.input.shape[:2], dtype = np.uint8)
        self.output = np.zeros(self.input.shape, np.uint8)


    def crop_to_alpha(self, img):
        x, y = self.alpha.nonzero()
        if len(x) == 0 or len(y) == 0: return img
        return img[np.min(x) : np.max(x), np.min(y) : np.max(y)]


    def save(self):
        # Apply alpha
        b, g, r, = cv.split(self.copy)
        img = cv.merge((b, g, r, self.alpha))

        cv.imwrite(self.outfile, self.crop_to_alpha(img))
        print('Saved')


    def segment(self):
        try:
            if self.rect_or_mask == 0:
                mask_type = cv.GC_INIT_WITH_RECT
                self.rect_or_mask = 1

            elif self.rect_or_mask == 1:
                mask_type = cv.GC_INIT_WITH_MASK

            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv.grabCut(self.copy, self.mask, self.rect, bgdmodel, fgdmodel, 1,
                       mask_type)

        except:
            import traceback
            traceback.print_exc()


    def load(self, nameFile):
        self.outfile = 'grabcut.png'
        filename = os.path.join('/home/truongdoan/dev/openCV/streamlit-hello', nameFile)
        # print()
        self.input    = cv.imread(filename)
        self.copy   = self.input.copy()         
        self.mask   = np.zeros(self.input.shape[:2], dtype = np.uint8)
        self.output = np.zeros(self.input.shape, np.uint8)
        self.alpha  = np.zeros(self.input.shape[:2], dtype = np.uint8)


    def run(self, nameFile):
        self.load(nameFile)
        self.reset()

        # Input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.input.shape[1] + 10, 90)

        # print('Draw a rectangle around the object using right mouse button')

        while True:
            cv.imshow('output', self.output)
            cv.imshow('input',  self.input)
            k = cv.waitKey(1)

            # Key bindings
            if k == 27 or k == ord('q'): break # exit
            elif k == ord('0'): self.value = self.DRAW_BG
            elif k == ord('1'): self.value = self.DRAW_FG
            elif k == ord('2'): self.value = self.DRAW_PR_BG
            elif k == ord('3'): self.value = self.DRAW_PR_FG
            elif k == ord('s'): self.save()
            elif k == ord('r'): self.reset()
            elif k == ord('n'): self.segment()
            else: continue

            self.alpha = np.where((self.mask == 1) + (self.mask == 3), 255,
                                  0).astype('uint8')
            img = cv.bitwise_and(self.copy, self.copy, mask = self.alpha)
            self.output = self.crop_to_alpha(img)


def grcut():
  img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
  
  if img is not None:
    imgg = Image.open(img)
    copy = np.asarray(imgg)
    drawling_mode = st.sidebar.selectbox("Drawing mode", ("rect", "transform", "point"))
    real_time_update = st.sidebar.checkbox("Real-time update", True)
    if drawling_mode == "point":
      point_display_radius = st.sidebar.slider("Point display radius", 1, 25, 3)
    canvas_rs = st_canvas(height=imgg.height, width=imgg.width,point_display_radius=point_display_radius if drawling_mode=='point' else None,display_toolbar=True,fill_color='',stroke_width=2, update_streamlit=True, background_image=imgg, drawing_mode=drawling_mode, stroke_color="red")
    form = st.form(key='form')
    # print(canvas_rs)
    rec = canvas_rs.json_data['objects']
    # for i in rec:
    #   if i['type'] == 'rect':
    #     x = i['left']
    #     y = i['top']
    #     w = i['width']
    #     h = i['height']
    #     cv.rectangle(copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
    max_one_rec = 0
    for i in range(len(rec)):
      if rec[i]['type'] == 'rect':
        max_one_rec += 1
        if max_one_rec > 1:
          st.warning("Only one rectangle is allowed")
          rec.pop(i)
    print(rec)
    st.image(copy, caption="Edit")
    submit = form.form_submit_button('Submit')
    print(os.getcwd())
    # st.write(img.getvalue())
    # App().run(img.name)
    # cv.destroyAllWindows()
grcut()