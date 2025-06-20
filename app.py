import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import tempfile
from yolov8_core import process_video
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import base64

st.set_page_config(layout="wide")
st.title("🚘 VelocIT: License Plate Detection")

uploaded_file = st.file_uploader("📤 Upload a traffic video", type=["mp4", "avi", "mov"])

roi_coords = None

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    st.markdown("### 🎞️ ")
    st.markdown(
        f"""
        <div style='max-width: 500px;'>
            <video controls style='width: 100%; height: auto;'>
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_video_path = tmp.name

    cap = cv2.VideoCapture(temp_video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        st.markdown("### 🖼️ Draw ROI (Region of Interest)")

        # Setting the width and height for canvas, reducing the canvas size
        canvas_width = 500  # Smaller canvas width
        canvas_height = int(image.height * (canvas_width / image.width))  # Adjust height based on width

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            left = int(obj["left"])
            top = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            roi_coords = (left, top, left + width, top + height)

    if st.button("▶️ Run Detection"):
        with st.spinner("Processing..."):
            output_path = process_video(temp_video_path, roi=roi_coords)

        st.success("✅ Done!")

        if output_path and os.path.exists(output_path):
            with open(output_path, "rb") as file:
                out_bytes = file.read()
            encoded_output = base64.b64encode(out_bytes).decode("utf-8")
            st.markdown("### 📽️ Output Video (smaller display)")
            st.markdown(
                f"""
                <div style='max-width: 500px;'>
                    <video controls style='width: 100%; height: auto;'>
                        <source src="data:video/mp4;base64,{encoded_output}" type="video/mp4">
                    </video>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("❌ Couldn't load output video.")
