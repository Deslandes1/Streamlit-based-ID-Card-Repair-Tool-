import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(page_title="ID Card Repair Tool", page_icon="🪪", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #e6f0fa; }
    .stButton>button { background-color: #4a8db7; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #2a6f96; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🪪 ID Card Repair Tool")
st.markdown("Upload a damaged ID card, mark the area, and repair or replace.")

# Initialize session state for canvas data
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = None
if "mask" not in st.session_state:
    st.session_state.mask = None

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** your ID card.
    2. Choose a **drawing mode**:
       - *Free draw* – paint over damaged areas (for inpainting).
       - *Rectangle* – draw a rectangle where you want the replacement image.
    3. If using *Rectangle*, upload the replacement image.
    4. Click **Repair**.
    5. Download the result.
    """)
    brush_size = st.slider("Brush Size", 1, 30, 10)
    drawing_mode = st.radio("Drawing Mode", ["Free draw", "Rectangle"])
    repair_mode = st.radio("Repair Method", ["Inpaint", "Replace with Image"])
    if repair_mode == "Replace with Image":
        replace_image_file = st.file_uploader("Upload replacement image", type=["jpg", "jpeg", "png"])
    else:
        replace_image_file = None
    uploaded_file = st.file_uploader("Upload ID card", type=["jpg", "jpeg", "png"])

# Main area
if uploaded_file is not None:
    # Read and resize image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Invalid image.")
        st.stop()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    MAX_DISPLAY_WIDTH = 500
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

    pil_image = Image.fromarray(image_rgb).convert('RGB')

    # Canvas drawing mode
    draw_mode = "freedraw" if drawing_mode == "Free draw" else "rect"
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✏️ Mark Area")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # transparent fill
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 0.8)",  # red for visibility
            background_image=pil_image,
            width=w,
            height=h,
            drawing_mode=draw_mode,
            update_streamlit=True,
            key="canvas",
        )

        # Store the canvas image data in session state
        if canvas_result is not None and canvas_result.image_data is not None:
            st.session_state.canvas_data = canvas_result.image_data.copy()
        else:
            st.session_state.canvas_data = None

    if st.button("🛠️ Repair", type="primary"):
        if st.session_state.canvas_data is None:
            st.warning("Please draw on the image first.")
        else:
            # Extract mask from alpha channel
            mask_data = st.session_state.canvas_data[:, :, 3].astype(np.uint8)
            mask = (mask_data > 0).astype(np.uint8) * 255

            # Resize mask to match image (just in case)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            st.session_state.mask = mask  # store for possible later use

            if np.sum(mask) == 0:
                st.warning("No area marked. Please draw.")
            else:
                with col2:
                    st.subheader("✅ Result")
                    if repair_mode == "Inpaint":
                        try:
                            repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
                            repaired_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
                            st.image(repaired_rgb, use_column_width=True)
                            # Download
                            repaired_pil = Image.fromarray(repaired_rgb)
                            buf = io.BytesIO()
                            repaired_pil.save(buf, format="PNG")
                            st.download_button("📥 Download", data=buf.getvalue(), file_name="repaired.png", mime="image/png")
                        except Exception as e:
                            st.error(f"Inpainting failed: {e}")
                    else:  # Replace with Image
                        if replace_image_file is None:
                            st.warning("Please upload a replacement image in the sidebar.")
                        else:
                            try:
                                rep_bytes = np.asarray(bytearray(replace_image_file.read()), dtype=np.uint8)
                                rep_img = cv2.imdecode(rep_bytes, cv2.IMREAD_COLOR)
                                if rep_img is None:
                                    st.error("Invalid replacement image.")
                                else:
                                    # For rectangle mode, use the mask's bounding box
                                    coords = cv2.findNonZero(mask)
                                    if coords is None:
                                        st.warning("Mask is empty. Draw a rectangle.")
                                    else:
                                        x, y, w_box, h_box = cv2.boundingRect(coords)
                                        # Clip to image boundaries
                                        x, y = max(0, x), max(0, y)
                                        w_box = min(w_box, w - x)
                                        h_box = min(h_box, h - y)
                                        if w_box <= 0 or h_box <= 0:
                                            st.warning("Drawn area too small or at edge.")
                                        else:
                                            # Resize replacement to fit the box
                                            rep_resized = cv2.resize(rep_img, (w_box, h_box))
                                            result_img = image.copy()
                                            result_img[y:y+h_box, x:x+w_box] = rep_resized
                                            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                            st.image(result_rgb, use_column_width=True)
                                            result_pil = Image.fromarray(result_rgb)
                                            buf = io.BytesIO()
                                            result_pil.save(buf, format="PNG")
                                            st.download_button("📥 Download", data=buf.getvalue(), file_name="replaced.png", mime="image/png")
                            except Exception as e:
                                st.error(f"Replacement failed: {e}")
else:
    st.info("👈 Upload an ID card image to begin.")
