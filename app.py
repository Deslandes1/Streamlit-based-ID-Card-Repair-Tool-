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

# Initialize session state variables to avoid "SessionInfo" errors
if "canvas_initialized" not in st.session_state:
    st.session_state.canvas_initialized = False
if "repair_done" not in st.session_state:
    st.session_state.repair_done = False

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** an image of your damaged ID card.
    2. **Draw** over the torn/missing areas using the brush.
    3. Click **Repair** to reconstruct the image.
    4. **Download** the repaired card.
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    brush_size = st.slider("Brush Size", 1, 30, 10)
    repair_mode = st.selectbox("Inpainting Method", ["Telea", "Navier-Stokes"])
    st.markdown("---")
    st.markdown("### 📁 Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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

        # Resize to a fixed display width (600px) to avoid cut-off
        MAX_DISPLAY_WIDTH = 600
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
            st.markdown("Use the brush to paint over damaged areas.")
            
            # Try to render the canvas; catch any session state errors
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
                st.error(f"Canvas initialization error: {e}. Please try reloading the page.")
                st.session_state.canvas_initialized = False
                canvas_result = None

        if st.button("🛠️ Repair", type="primary"):
            if not st.session_state.canvas_initialized or canvas_result is None:
                st.warning("Please wait for the canvas to load, then draw on the image.")
            elif canvas_result.image_data is None:
                st.warning("No drawing data found. Please draw on the image first.")
            else:
                # Extract mask from alpha channel
                mask_data = canvas_result.image_data[:, :, 3].astype(np.uint8)
                mask = (mask_data > 0).astype(np.uint8) * 255

                if np.sum(mask) == 0:
                    st.warning("You haven't marked any damaged areas. Draw with the brush first.")
                else:
                    try:
                        inpaint_method = cv2.INPAINT_TELEA if repair_mode == "Telea" else cv2.INPAINT_NS
                        repaired = cv2.inpaint(image, mask, 3, inpaint_method)
                        repaired_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

                        with col2:
                            st.subheader("✅ Repaired Card")
                            st.image(repaired_rgb, use_container_width=True)

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
                            st.session_state.repair_done = True
                    except Exception as e:
                        st.error(f"Repair failed: {e}. Please try again or adjust the mask.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}. Please try again with a different image.")
else:
    st.info("👈 Upload an image to get started.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OpenCV, and streamlit-drawable-canvas.")
