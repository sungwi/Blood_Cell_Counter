import streamlit as st
from PIL import Image

from cell_count import *

def main():
    st.title("Blood Cell Counter")

    menu = ["YOLOv8n Base", "YOLOv8n Extended", "U-NET Base", "U-NET Extended"]
    
    choice = st.sidebar.selectbox("Choose an option", menu)
    
    if choice == "YOLOv8n Base":
        yolo_base()
    elif choice == "YOLOv8n Extended":
        yolo_extended()
    elif choice == "U-NET Base":
        unet_base()
    elif choice == "U-NET Extended":
        unet_extended()


def yolo_base():
    st.subheader("YOLOv8n base model: Trained on 100 medical images")
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
    st.subheader("YOLOv8n extended model: Trained on 366 medical images")
    uploaded_file = st.file_uploader("Upload a blood cell image (JPEG or PNG)", type=["jpeg", "png", "jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        processed_image, cell_count = detect_cells_yolo_extended(image)
        with col2:
            st.image(processed_image, caption  = "Processed Image", use_container_width=True)

        st.text(f"Total Cells Detected: {cell_count}")

def unet_base():
    st.subheader("UNET baseline model: Trained on 100 emdical images")
    uploaded_file = st.file_uploader("Upload a blood cell image (JPEG or PNG)", type=["jpeg", "png", "jpg"])
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True, channels="BGR")
        processed_image, cell_count = detect_cells_unet_baseline(image)
        with col2:
            st.image(processed_image, caption="Processed Image", use_container_width=True, channels="BGR")

        st.text(f"Total Cells Detected: {cell_count}")

def unet_extended():
    st.subheader("UNET extended model: Trained on 366 medical images")
    uploaded_file = st.file_uploader("Upload a blood cell image (JPEG or PNG)", type=["jpeg", "png", "jpg"])
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True, channels="BGR")
        processed_image, cell_count = detect_cells_unet_extended(image)
        with col2:
            st.image(processed_image, caption="Processed Image", use_container_width=True, channels="BGR")

        st.text(f"Total Cells Detected: {cell_count}")

if __name__ == "__main__":
    main()