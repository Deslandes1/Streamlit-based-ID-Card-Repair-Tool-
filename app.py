import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io

# Page configuration
st.set_page_config(
    page_title="ID Card Repair Tool",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light blue theme
st.markdown("""
<style>
    .stApp { background-color: #e6f0fa; }
    .css-1d391kg { background-color: #d4e6f5; }
    .sidebar .sidebar-content { background-color: #c8dff5; }
    .stButton>button { background-color: #4a8db7; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #2a6f96; color: white; }
    h1, h2, h3 { color: #1a3b5a; }
    .stMarkdown { color: #1a3b5a; }
</style>
""", unsafe_allow_html=True)

st.title("🪪 ID Card Repair Tool")
st.markdown("Upload a damaged ID card, mark the broken areas, and we'll repair it automatically.")

# Initialize session state
if "canvas_initialized" not in st.session_state:
    st.session_state.canvas_initialized = False

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** your ID card image.
    2. **Draw** over the torn/missing area with the brush.
    3. Choose a **repair mode**:
       - *Inpaint* – fill with surrounding pixels.
       - *Replace with Image* – overlay a new photo on the drawn region.
    4. If using *Replace*, upload the replacement image.
    5. Click **Repair**.
    6. **Download** the result.
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    brush_size = st.slider("Brush Size", 1, 30, 10)
    repair_mode = st.radio("Repair Method", ["Inpaint", "Replace with Image"])
    if repair_mode == "Replace with Image":
        replace_image_file = st.file_uploader("Upload replacement image (e.g., your photo)", type=["jpg", "jpeg", "png"])
    else:
        replace_image_file = None
    st.markdown("---")
    st.markdown("### 📁 Upload ID Card")
    uploaded_file = st.file_uploader("Choose ID card image...", type=["jpg", "jpeg", "png"])

# Main area
if uploaded_file is not None:
    try:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not decode the image. Please upload a valid JPG or PNG file.")
            st.stop()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Resize to a fixed display width (500px) – fits well in a column
        MAX_DISPLAY_WIDTH = 500
        if w > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb).convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✏️ Mark Damage")
            st.markdown("Use the brush to paint over the missing/damaged area.")
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=brush_size,
                    stroke_color="rgba(255, 255, 255, 1)",
                    background_image=pil_image,
                    width=w,
                    height=h,
                    update_streamlit=True,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                st.session_state.canvas_initialized = True
            except Exception as e:
                st.error(f"Canvas error: {e}. Please reload the page.")
                canvas_result = None

        if st.button("🛠️ Repair", type="primary"):
            if not st.session_state.canvas_initialized or canvas_result is None:
                st.warning("Canvas not ready. Please wait and try again.")
            elif canvas_result.image_data is None:
                st.warning("No drawing data. Please draw on the image first.")
            else:
                # Extract mask from alpha channel
                mask_data = canvas_result.image_data[:, :, 3].astype(np.uint8)
                mask = (mask_data > 0).astype(np.uint8) * 255

                # *** CRITICAL FIX: Ensure mask matches image dimensions ***
                if mask.shape[:2] != (h, w):
                    # Resize mask to match the image
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                if np.sum(mask) == 0:
                    st.warning("You haven't marked any area. Draw with the brush first.")
                else:
                    with col2:
                        st.subheader("✅ Result")
                        # Process based on mode
                        if repair_mode == "Inpaint":
                            try:
                                inpaint_method = cv2.INPAINT_TELEA  # Telea is default
                                repaired = cv2.inpaint(image, mask, 3, inpaint_method)
                                repaired_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
                                st.image(repaired_rgb, use_column_width=True)
                                # Download
                                repaired_pil = Image.fromarray(repaired_rgb)
                                buf = io.BytesIO()
                                repaired_pil.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                st.download_button(
                                    label="📥 Download Repaired Image",
                                    data=byte_im,
                                    file_name="repaired_id_card.png",
                                    mime="image/png",
                                )
                            except Exception as e:
                                st.error(f"Inpainting failed: {e}")
                        else:  # Replace with Image
                            if replace_image_file is None:
                                st.warning("Please upload a replacement image in the sidebar.")
                            else:
                                try:
                                    # Read replacement image
                                    rep_bytes = np.asarray(bytearray(replace_image_file.read()), dtype=np.uint8)
                                    rep_img = cv2.imdecode(rep_bytes, cv2.IMREAD_COLOR)
                                    if rep_img is None:
                                        st.error("Could not decode replacement image.")
                                    else:
                                        # Compute bounding box of the mask
                                        coords = cv2.findNonZero(mask)
                                        if coords is None:
                                            st.warning("Mask is empty. Draw again.")
                                        else:
                                            x, y, w_box, h_box = cv2.boundingRect(coords)
                                            # Ensure box is within image bounds
                                            x = max(0, x)
                                            y = max(0, y)
                                            w_box = min(w_box, w - x)
                                            h_box = min(h_box, h - y)
                                            if w_box <= 0 or h_box <= 0:
                                                st.warning("Drawn area is too small or at the edge.")
                                            else:
                                                # Resize replacement image to fit the box
                                                rep_resized = cv2.resize(rep_img, (w_box, h_box))
                                                # Overlay on original image
                                                result_img = image.copy()
                                                result_img[y:y+h_box, x:x+w_box] = rep_resized
                                                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                                st.image(result_rgb, use_column_width=True)
                                                # Download
                                                result_pil = Image.fromarray(result_rgb)
                                                buf = io.BytesIO()
                                                result_pil.save(buf, format="PNG")
                                                byte_im = buf.getvalue()
                                                st.download_button(
                                                    label="📥 Download Result",
                                                    data=byte_im,
                                                    file_name="replaced_id_card.png",
                                                    mime="image/png",
                                                )
                                except Exception as e:
                                    st.error(f"Replacement failed: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}. Please try again.")
else:
    st.info("👈 Upload an ID card image to get started.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OpenCV, and streamlit-drawable-canvas.")
