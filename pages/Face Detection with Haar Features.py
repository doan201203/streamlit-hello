from __future__ import print_function
import streamlit as st
import cv2 as cv
import os
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.set_page_config(page_title="Face Detection with Haar Features")