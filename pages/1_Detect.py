import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.request import urlopen
import json
import PIL
import os
import sys
from ultralytics import YOLO

# Add the path to the parent folder of the current script's directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import settings

model_path = settings.DETECTION_MODEL
# Путь к папке с изображениями
image_folder = settings.IMAGES_DIR

# Получение списка имен файлов изображений
image_files = ['Upload image...'] + os.listdir(image_folder)

# Setting page layout
st.set_page_config(
    page_title="Well detection using YOLOv8",  # Setting page title
    page_icon="💩",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 1, 100, 25)) / 100
    nms = float(st.slider(
        "Select Model IOU", 1, 100, 70)) / 100

# Creating main page heading
st.title("Well Detection using YOLOv8")

# Создание выпадающего списка для выбора изображения
selected_image_file = st.selectbox(
    'Upload image or select preloadeded image', image_files)

# Загрузка и отображение выбранного изображения
if selected_image_file == 'Upload image...':
    uploaded_file = st.file_uploader(
        "Upload a satellite image", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    if uploaded_file is not None:
        # Отображение загруженного изображения
        source_img = uploaded_file
    else:
        source_img = None
else:
    # Загрузка и отображение выбранного изображения
    source_img = os.path.join(image_folder, selected_image_file)
    # source_img = PIL.Image.open(image_path)
    # Дальнейшая обработка выбранного изображения...

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)

        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        resized_image = PIL.ImageOps.autocontrast(
            uploaded_image.resize((640, 640)))

try:
    model = YOLO(model_path)
    # st.write("Model loaded successfully!")
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image,
                        conf=confidence,
                        iou=nms
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption=f'Detected Image - {len(boxes)} WELLS',
                 use_column_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
