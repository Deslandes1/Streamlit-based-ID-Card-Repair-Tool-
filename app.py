import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from streamlit_drawable_canvas import st_canvas
import io
import hashlib

# ---------- Helper functions ----------
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

def detect_and_straighten_card(image):
    """Detect the card's edges and apply perspective transform to straighten it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # no contours found
    # Get largest contour (assuming it's the card)
    cnt = max(contours, key=cv2.contourArea)
    # Approximate polygon
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    # If we have 4 points, we can straighten
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # top-right
        rect[3] = pts[np.argmax(diff)] # bottom-left
        # Compute width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    else:
        return image  # could not find 4 corners

def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = correct_image_orientation(image)
    return image

# ---------- Page config ----------
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

# ---------- Session state ----------
if "processed" not in st.session_state:
    st.session_state.processed = None          # final processed image (BGR) – after straighten
    st.session_state.original = None           # original image after orientation (BGR)
    st.session_state.image_rgb = None
    st.session_state.h = 0
    st.session_state.w = 0
    st.session_state.pil_image = None
    st.session_state.canvas_data = None
    st.session_state.mask = None
    st.session_state.uploaded_file_hash = None
    st.session_state.repaired = None           # store repaired image for display

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** your ID card.
    2. Optionally enable **Straighten Card** to fix leaning.
    3. Choose a **drawing mode** (Free draw or Rectangle).
    4. If using *Replace*, upload a replacement image.
    5. Draw on the card.
    6. Click **Repair**.
    7. **Download** the result.
    """)
    uploaded_file = st.file_uploader("Upload ID card", type=["jpg", "jpeg", "png"])
    straighten = st.checkbox("🔄 Straighten Card (fix leaning)", value=True)
    brush_size = st.slider("Brush Size", 1, 30, 10)
    drawing_mode = st.radio("Drawing Mode", ["Free draw", "Rectangle"])
    repair_mode = st.radio("Repair Method", ["Inpaint", "Replace with Image"])
    replace_image_file = None
    if repair_mode == "Replace with Image":
        replace_image_file = st.file_uploader("Upload replacement image", type=["jpg", "jpeg", "png"])
    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------- Process uploaded file ----------
if uploaded_file is not None:
    # Check if new file
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.uploaded_file_hash != file_hash:
        # Read image
        img = read_uploaded_file(uploaded_file)
        if img is not None:
            st.session_state.original = img
            # Straighten if enabled
            if straighten:
                img = detect_and_straighten_card(img)
            st.session_state.processed = img
            st.session_state.image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.session_state.h, st.session_state.w, _ = img.shape
            # Resize for display if needed
            MAX_DISPLAY = 500
            if st.session_state.w > MAX_DISPLAY:
                scale = MAX_DISPLAY / st.session_state.w
                new_w = int(st.session_state.w * scale)
                new_h = int(st.session_state.h * scale)
                st.session_state.processed = cv2.resize(st.session_state.processed, (new_w, new_h))
                st.session_state.image_rgb = cv2.cvtColor(st.session_state.processed, cv2.COLOR_BGR2RGB)
                st.session_state.h, st.session_state.w, _ = st.session_state.processed.shape
            st.session_state.pil_image = Image.fromarray(st.session_state.image_rgb).convert('RGB')
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.canvas_data = None
            st.session_state.mask = None
            st.session_state.repaired = None
        else:
            st.error("Invalid image file.")
else:
    # Clear session if no file (optional)
    if st.session_state.uploaded_file_hash is not None:
        for key in ['original', 'processed', 'image_rgb', 'pil_image', 'canvas_data', 'mask', 'repaired']:
            if key in st.session_state:
                st.session_state[key] = None
        st.session_state.uploaded_file_hash = None
        st.session_state.h = st.session_state.w = 0

# ---------- Main display ----------
if st.session_state.processed is not None and st.session_state.pil_image is not None:
    h, w = st.session_state.h, st.session_state.w
    col1, col2 = st.columns(2)
    draw_mode = "freedraw" if drawing_mode == "Free draw" else "rect"

    with col1:
        st.subheader("✏️ Mark Area")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 0.8)",
            background_image=st.session_state.pil_image,
            width=w,
            height=h,
            drawing_mode=draw_mode,
            update_streamlit=True,
            key="canvas",
        )
        if canvas_result is not None and canvas_result.image_data is not None:
            st.session_state.canvas_data = canvas_result.image_data.copy()

    if st.button("🛠️ Repair", type="primary"):
        if st.session_state.canvas_data is None:
            st.warning("Please draw on the image first.")
        else:
            # Extract mask
            mask_data = st.session_state.canvas_data[:, :, 3].astype(np.uint8)
            mask = (mask_data > 0).astype(np.uint8) * 255
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            st.session_state.mask = mask

            if np.sum(mask) == 0:
                st.warning("No area marked. Please draw.")
            else:
                image = st.session_state.processed
                if repair_mode == "Inpaint":
                    try:
                        repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
                        st.session_state.repaired = repaired
                    except Exception as e:
                        st.error(f"Inpainting failed: {e}")
                        st.session_state.repaired = None
                else:  # Replace
                    if replace_image_file is None:
                        st.warning("Please upload a replacement image.")
                        st.session_state.repaired = None
                    else:
                        try:
                            rep_bytes = np.asarray(bytearray(replace_image_file.read()), dtype=np.uint8)
                            rep_img = cv2.imdecode(rep_bytes, cv2.IMREAD_COLOR)
                            if rep_img is None:
                                st.error("Invalid replacement image.")
                                st.session_state.repaired = None
                            else:
                                rep_img = correct_image_orientation(rep_img)
                                coords = cv2.findNonZero(mask)
                                if coords is None:
                                    st.warning("Mask is empty. Draw a rectangle.")
                                    st.session_state.repaired = None
                                else:
                                    x, y, w_box, h_box = cv2.boundingRect(coords)
                                    x, y = max(0, x), max(0, y)
                                    w_box = min(w_box, w - x)
                                    h_box = min(h_box, h - y)
                                    if w_box <= 0 or h_box <= 0:
                                        st.warning("Drawn area too small or at edge.")
                                        st.session_state.repaired = None
                                    else:
                                        rep_resized = cv2.resize(rep_img, (w_box, h_box))
                                        result_img = image.copy()
                                        result_img[y:y+h_box, x:x+w_box] = rep_resized
                                        st.session_state.repaired = result_img
                        except Exception as e:
                            st.error(f"Replacement failed: {e}")
                            st.session_state.repaired = None

    # Display result in col2 if exists
    with col2:
        st.subheader("✅ Result")
        if st.session_state.repaired is not None:
            repaired_rgb = cv2.cvtColor(st.session_state.repaired, cv2.COLOR_BGR2RGB)
            st.image(repaired_rgb, use_column_width=True)
            repaired_pil = Image.fromarray(repaired_rgb)
            buf = io.BytesIO()
            repaired_pil.save(buf, format="PNG")
            st.download_button("📥 Download", data=buf.getvalue(), file_name="repaired.png", mime="image/png")
        else:
            st.info("Click 'Repair' to see the result.")

else:
    st.info("👈 Upload an ID card image to begin.")
