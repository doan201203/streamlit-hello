import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from my_utils.cbri import CBRI

st.title('Instance Search')
a = CBRI()
print(a.kmeans)
st.write('1. Dataset')

st.write('2. Methods')
st.write('3. Evaluation')
st.write('4. Results')
