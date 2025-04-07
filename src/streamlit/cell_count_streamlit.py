import streamlit as st
from PIL import Image

from cell_count import *

def main():
    st.title("Blood Cell Counter")

    menu = ["YOLO Base", "YOLO Extended", "U-NET Base", "U-NET Extended"]
    
    choice = st.sidebar.selectbox("Choose an option", menu)
    
    if choice == "YOLO Base":
        yolo_base()
    elif choice == "YOLO Extended":
        yolo_extended()
    elif choice == "U-NET Base":
        unet_base()
    elif choice == "U-NET Extended":
        unet_extended()


def yolo_base():
    st.subheader("YOLO model: Trained on")
    uploaded_file = st.file_uploader("Upload a blood cell image (JPEG or PNG)", type=["jpeg", "png", "jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        processed_image, cell_count = detect_cells_yolo_baseline(image)
        with col2:
            st.image(processed_image, caption="Processed Image", use_container_width=True)

        st.text(f"Total Cells Detected: {cell_count}")

def yolo_extended():
    cell_count = -1
    st.text(f"Total Cells Detected: {cell_count}")

def unet_base():
    cell_count = -1
    st.text(f"Total Cells Detected: {cell_count}")

def unet_extended():
    cell_count = -1
    st.text(f"Total Cells Detected: {cell_count}")
    
if __name__ == "__main__":
    main()