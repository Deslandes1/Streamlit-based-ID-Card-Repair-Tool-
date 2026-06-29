import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from streamlit_drawable_canvas import st_canvas
import io
import hashlib

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def correct_image_orientation(image):
    """Correct image orientation using EXIF data."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        exif = pil_img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    orientation = value
                    if orientation == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif orientation == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif orientation == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
    except:
        pass
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def auto_straighten(image):
    """
    Attempt to automatically deskew the image by finding the largest quadrilateral.
    Assumes the ID card is the main object.
    """
    # Safety check: if image is None or too small, return as is.
    if image is None or image.shape[0] < 10 or image.shape[1] < 10:
        return image

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        # Find the largest contour by area
        largest = max(contours, key=cv2.contourArea)
        # If contour area is too small, skip
        if cv2.contourArea(largest) < 100:
            return image
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) != 4:
            return image
        pts = approx.reshape(4, 2)
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # top-right
        rect[3] = pts[np.argmax(diff)] # bottom-left
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth < 10 or maxHeight < 10:
            return image
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    except Exception as e:
        # If anything fails, return the original image
        return image

def process_uploaded_file(uploaded_file):
    """Read and process uploaded file, correct orientation, and optionally straighten."""
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = correct_image_orientation(image)
    return image

# ------------------------------------------------------------
# Page config and styling
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "image" not in st.session_state:
    st.session_state.image = None
    st.session_state.image_rgb = None
    st.session_state.h = 0
    st.session_state.w = 0
    st.session_state.pil_image = None
    st.session_state.canvas_data = None
    st.session_state.mask = None
    st.session_state.uploaded_file_hash = None
    st.session_state.repair_trigger = False

# ------------------------------------------------------------
# Sidebar widgets
# ------------------------------------------------------------
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** your ID card (auto‑rotated & auto‑straightened).
    2. Choose **Drawing Mode**:
       - *Free draw* – paint over damaged areas.
       - *Rectangle* – draw a rectangle for replacement.
    3. Choose **Repair Method**.
    4. Click **Repair**.
    5. **Download** the result.
    """)
    brush_size = st.slider("Brush Size", 1, 30, 10, key="brush_size")
    drawing_mode = st.radio("Drawing Mode", ["Free draw", "Rectangle"], key="drawing_mode")
    repair_mode = st.radio("Repair Method", ["Inpaint", "Replace with Image"], key="repair_mode")
    auto_straighten = st.checkbox("Auto‑straighten (fix leaning frame)", value=True, key="auto_straighten")
    if repair_mode == "Replace with Image":
        replace_image_file = st.file_uploader("Upload replacement image", type=["jpg", "jpeg", "png"], key="replace_image")
    else:
        replace_image_file = None
    uploaded_file = st.file_uploader("Upload ID card", type=["jpg", "jpeg", "png"], key="upload_id")

    if st.button("🔄 Reset All", key="reset_all"):
        st.session_state.image = None
        st.session_state.image_rgb = None
        st.session_state.pil_image = None
        st.session_state.canvas_data = None
        st.session_state.mask = None
        st.session_state.uploaded_file_hash = None
        st.session_state.repair_trigger = False
        st.rerun()

# ------------------------------------------------------------
# Process uploaded image
# ------------------------------------------------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.uploaded_file_hash != file_hash:
        image = process_uploaded_file(uploaded_file)
        if image is not None:
            # Auto‑straighten if enabled
            if auto_straighten:
                image = auto_straighten(image)
            st.session_state.image = image
            st.session_state.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.h, st.session_state.w, _ = image.shape
            # Resize for display
            MAX_DISPLAY_WIDTH = 500
            if st.session_state.w > MAX_DISPLAY_WIDTH:
                scale = MAX_DISPLAY_WIDTH / st.session_state.w
                new_w = int(st.session_state.w * scale)
                new_h = int(st.session_state.h * scale)
                st.session_state.image = cv2.resize(st.session_state.image, (new_w, new_h))
                st.session_state.image_rgb = cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB)
                st.session_state.h, st.session_state.w, _ = st.session_state.image.shape
            st.session_state.pil_image = Image.fromarray(st.session_state.image_rgb).convert('RGB')
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.canvas_data = None
            st.session_state.mask = None
            st.session_state.repair_trigger = False
        else:
            st.error("Invalid image file.")
else:
    # Clear stale state if no file
    if st.session_state.image is not None:
        st.session_state.image = None
        st.session_state.image_rgb = None
        st.session_state.pil_image = None
        st.session_state.canvas_data = None
        st.session_state.mask = None
        st.session_state.uploaded_file_hash = None
        st.session_state.repair_trigger = False

# ------------------------------------------------------------
# Main display and repair
# ------------------------------------------------------------
if st.session_state.image is not None:
    h, w = st.session_state.h, st.session_state.w
    pil_image = st.session_state.pil_image

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
            key="repair_canvas",
        )

        if canvas_result is not None and canvas_result.image_data is not None:
            st.session_state.canvas_data = canvas_result.image_data.copy()

    # Repair button
    if st.button("🛠️ Repair", key="repair_btn", type="primary"):
        st.session_state.repair_trigger = True

    # Process repair only when triggered
    if st.session_state.repair_trigger and st.session_state.canvas_data is not None:
        # Extract mask
        mask_data = st.session_state.canvas_data[:, :, 3].astype(np.uint8)
        mask = (mask_data > 0).astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        st.session_state.mask = mask

        if np.sum(mask) == 0:
            st.warning("No area marked. Please draw.")
            st.session_state.repair_trigger = False
        else:
            with col2:
                st.subheader("✅ Result")
                image = st.session_state.image
                if repair_mode == "Inpaint":
                    try:
                        repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
                        repaired_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
                        st.image(repaired_rgb, use_container_width=True)
                        repaired_pil = Image.fromarray(repaired_rgb)
                        buf = io.BytesIO()
                        repaired_pil.save(buf, format="PNG")
                        st.download_button("📥 Download Repaired", data=buf.getvalue(),
                                           file_name="repaired.png", mime="image/png", key="dl_repair")
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
                                        st.image(result_rgb, use_container_width=True)
                                        result_pil = Image.fromarray(result_rgb)
                                        buf = io.BytesIO()
                                        result_pil.save(buf, format="PNG")
                                        st.download_button("📥 Download Replaced", data=buf.getvalue(),
                                                           file_name="replaced.png", mime="image/png", key="dl_replace")
                        except Exception as e:
                            st.error(f"Replacement failed: {e}")
                # Reset trigger so the repair runs only once per click
                st.session_state.repair_trigger = False
    elif st.session_state.repair_trigger and st.session_state.canvas_data is None:
        st.warning("Please draw on the image first.")
        st.session_state.repair_trigger = False

else:
    st.info("👈 Upload an ID card image to begin.")
