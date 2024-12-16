import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from miscs.image_editor.Image_Edit import ImageProcessor

# Set the page configuration for the Streamlit app
st.set_page_config(page_title="Image Editor", page_icon=":camera:", layout="wide")

# Display a main header for the app
st.markdown("# Image Editor")

# Display a sidebar header for the app
st.sidebar.header("Image Editor")

# Create a file uploader widget in the sidebar
# This allows users to upload an image (PNG, JPG, or JPEG)
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

# Function to create an ImageProcessor object
def load_tool(img):
    return ImageProcessor(img)

# Function to convert an OpenCV image (NumPy array) to a PIL Image object
def cv2_to_pil(cv2_img):
    """Converts an OpenCV image (NumPy array) to a PIL Image object."""
    pil_img = Image.fromarray(cv2_img)
    return pil_img

# Check if a file has been uploaded
if file is not None:
    # Open the uploaded image using Pillow and convert it to a NumPy array
    img = Image.open(file)
    img = np.array(img)
    # Create an ImageProcessor object with the loaded image
    extractor = load_tool(img)

    # Check if the image is being loaded for the first time
    if "img_name" not in st.session_state:
        # Initialize session state variables for the first image upload
        st.session_state['img_name'] = file
        st.session_state['extractor'] = extractor
        st.session_state['prev_img'] = None
        st.session_state['prev_name'] = file.name
        if 'my_canvas' in st.session_state:
            del st.session_state['my_canvas']
    else:
        # If a different image is uploaded, update the session state variables
        if st.session_state['img_name'] != file:
            st.session_state['img_name'] = file
            st.session_state['extractor'] = extractor
        if st.session_state['prev_name'] in st.session_state:
            del st.session_state[st.session_state['prev_name']]
        st.session_state['prev_img'] = None
        st.session_state['prev_name'] = file.name

    # Display a header for the image tools section
    st.markdown("## Tools")
    # Create a radio button to select an image editing tool
    tool = st.radio("Select a tool", ["Flip", "Rotation", "Colorspace", "Translation", "Cropping"], horizontal=True)

    # Calculate the scale for displaying the image
    scale = st.session_state['extractor'].calculate_scale()
    # Resize the image based on the calculated scale
    img = st.session_state['extractor'].resize_image(scale)
    # Create two columns for layout purposes
    col = st.columns(2)
    # If the tool is not 'Cropping', display the original image in the first column
    if tool != "Cropping":
        col[0].image(img, caption="Original Image")

    # --- Image Tool Implementations ---

    # Flip Tool
    if tool == "Flip":
        # Display instructions on how to use the Flip tool (Vietnamese)
        st.write("### Hướng dẫn công cụ Lật:")
        st.write("- Chọn `direction` (Ngang hoặc Dọc).")
        st.write("- Nhấn nút `Apply` để lật ảnh.")
        # Create a radio button to choose the flip direction (horizontal or vertical)
        direction = st.radio("Select direction", ["Horizontal", "Vertical"], key="flip")
        # When the "Apply" button is clicked
        if st.button("Apply"):
            # Flip the image based on the selected direction using the ImageProcessor
            img = st.session_state['extractor'].flip(direction.lower()).image
            # Rerun the app to update the image
            st.rerun()
    # Rotation Tool
    elif tool == "Rotation":
        # Display instructions on how to use the Rotation tool (Vietnamese)
        st.write("### Hướng dẫn công cụ Xoay:")
        st.write("- Sử dụng `slider` để chọn `angle` xoay (từ -180 đến 180 độ).")
        st.write("- Nhấn nút `Apply` để xoay ảnh.")
        # Create a slider to select the rotation angle (from -180 to 180 degrees)
        angle = st.slider("Select angle", -180, 180, 0, 1)
        # When the "Apply" button is clicked
        if st.button("Apply"):
            # Rotate the image based on the selected angle using the ImageProcessor
            img = st.session_state['extractor'].rotate(angle).image
            # Rerun the app to update the image
            st.rerun()
    # Colorspace Tool
    elif tool == "Colorspace":
        # Display instructions on how to use the Colorspace tool (Vietnamese)
        st.write("### Hướng dẫn công cụ Không gian màu:")
        st.write("- Chọn `colorspace` mong muốn từ menu thả xuống (RGB, HSV hoặc Gray).")
        st.write("- Nhấn nút `Apply` để thay đổi không gian màu của ảnh.")
        # Create a select box to choose the target colorspace (RGB, HSV, or Gray)
        target_space = st.selectbox("Select colorspace", ["RGB", "HSV", "Gray"], key="colorspace")
        # When the "Apply" button is clicked
        if st.button("Apply"):
            # Change the image's colorspace based on the selected target_space using the ImageProcessor
            img = st.session_state['extractor'].change_colorspace(target_space).image
            # Rerun the app to update the image
            st.rerun()
    # Translation Tool
    elif tool == "Translation":
         # Display instructions on how to use the Translation tool (Vietnamese)
        st.write("### Hướng dẫn công cụ Dịch chuyển:")
        st.write("- Sử dụng `sliders` để đặt `x offset` và `y offset`.")
        st.write("- Nhấn nút `Apply` để dịch chuyển ảnh.")
        # Create sliders to select the x and y offset for translation
        x_offset = st.slider("Select x offset", -100, 100, 0, 1)
        y_offset = st.slider("Select y offset", -100, 100, 0, 1)
         # When the "Apply" button is clicked
        if st.button("Apply"):
            # Translate the image based on the selected x and y offsets using the ImageProcessor
            img = st.session_state['extractor'].translate(x_offset, y_offset).image
            # Rerun the app to update the image
            st.rerun()
    # Cropping Tool
    elif tool == "Cropping":
        # Display instructions on how to use the Cropping tool (Vietnamese)
        st.write("### Hướng dẫn công cụ Cắt:")
        st.write("- Vẽ một hình chữ nhật để cắt ảnh.")
        st.write("- Nhấn nút `Submit` để cắt ảnh dựa trên hình chữ nhật đã vẽ.")
        # Use a Streamlit form to manage the canvas and cropping
        with st.form(key="form", clear_on_submit=True):
            # Create a drawing canvas for cropping
            # The canvas allows users to draw a rectangle on top of the image
            col[0] = st_canvas(
                background_color="white",
                background_image=cv2_to_pil(st.session_state['extractor'].get_img()),
                display_toolbar=True,
                height=img.shape[0],
                width=img.shape[1],
                drawing_mode='rect',
                stroke_width=2,
                fill_color="",
                stroke_color="red",
            )
            # Create a submit button within the form
            submit = st.form_submit_button("Submit")
            # If the submit button is clicked
            if submit:
                # Check if there is any JSON data from the canvas (i.e., a rectangle has been drawn)
                if col[0].json_data is not None:
                    # Get the rectangle data from the JSON
                    rec = col[0].json_data['objects']
                     # Extract the coordinates and dimensions of the drawn rectangle
                    x = rec[0]['left']
                    y = rec[0]['top']
                    w = rec[0]['width']
                    h = rec[0]['height']
                    # Crop the image using the ImageProcessor based on the extracted data
                    img = st.session_state['extractor'].crop(x, y, x + w, y + h).image
                    # Rerun the app to update the image
                    st.rerun()
                # If no rectangle is drawn
                else:
                    # Display a warning message
                    st.warning("Vui lòng vẽ một hình chữ nhật để cắt ảnh")