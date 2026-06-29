import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from streamlit_drawable_canvas import st_canvas
import io

def correct_image_orientation(image):
    """
    Corrects image orientation based on EXIF data.
    image: numpy array (BGR) from OpenCV.
    Returns: correctly oriented numpy array (BGR).
    """
    # Convert to PIL for EXIF handling
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        # Get EXIF orientation
        exif = pil_img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    orientation = value
                    if orientation == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif orientation == 6:
                        pil_img = pil_img.rotate(270, expand=True)  # 270° clockwise (or 90° counter‑clockwise)
                    elif orientation == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
    except Exception as e:
        # If EXIF reading fails, just return original
        pass
    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Page config
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

# Session state for canvas persistence
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
        replace_image_file = st.file_uploader("Upload replacement image (e.g., your photo)", type=["jpg", "jpeg", "png"])
    else:
        replace_image_file = None
    uploaded_file = st.file_uploader("Upload ID card", type=["jpg", "jpeg", "png"])

# Main area
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Invalid image.")
        st.stop()
    # Correct orientation
    image = correct_image_orientation(image)

    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Resize to fit column
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
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 0.8)",
            background_image=pil_image,
            width=w,
            height=h,
            drawing_mode=draw_mode,
            update_streamlit=True,
            key="canvas",
        )

        if canvas_result is not None and canvas_result.image_data is not None:
            st.session_state.canvas_data = canvas_result.image_data.copy()
        else:
            st.session_state.canvas_data = None

    if st.button("🛠️ Repair", type="primary"):
        if st.session_state.canvas_data is None:
            st.warning("Please draw on the image first.")
        else:
            mask_data = st.session_state.canvas_data[:, :, 3].astype(np.uint8)
            mask = (mask_data > 0).astype(np.uint8) * 255
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            st.session_state.mask = mask

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
                            repaired_pil = Image.fromarray(repaired_rgb)
                            buf = io.BytesIO()
                            repaired_pil.save(buf, format="PNG")
                            st.download_button("📥 Download", data=buf.getvalue(), file_name="repaired.png", mime="image/png")
                        except Exception as e:
                            st.error(f"Inpainting failed: {e}")
                    else:  # Replace with Image
                        if replace_image_file is None:
                            st.warning("Please upload a replacement image.")
                        else:
                            try:
                                rep_bytes = np.asarray(bytearray(replace_image_file.read()), dtype=np.uint8)
                                rep_img = cv2.imdecode(rep_bytes, cv2.IMREAD_COLOR)
                                if rep_img is None:
                                    st.error("Invalid replacement image.")
                                else:
                                    # Also correct orientation of replacement image
                                    rep_img = correct_image_orientation(rep_img)
                                    coords = cv2.findNonZero(mask)
                                    if coords is None:
                                        st.warning("Mask is empty. Draw a rectangle.")
                                    else:
                                        x, y, w_box, h_box = cv2.boundingRect(coords)
                                        x, y = max(0, x), max(0, y)
                                        w_box = min(w_box, w - x)
                                        h_box = min(h_box, h - y)
                                        if w_box <= 0 or h_box <= 0:
                                            st.warning("Drawn area too small or at edge.")
                                        else:
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
