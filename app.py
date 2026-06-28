import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import tempfile
import os

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
    .stApp {
        background-color: #e6f0fa;
    }
    .css-1d391kg {
        background-color: #d4e6f5;
    }
    .sidebar .sidebar-content {
        background-color: #c8dff5;
    }
    .stButton>button {
        background-color: #4a8db7;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #2a6f96;
        color: white;
    }
    h1, h2, h3 {
        color: #1a3b5a;
    }
    .stMarkdown {
        color: #1a3b5a;
    }
</style>
""", unsafe_allow_html=True)

st.title("🪪 ID Card Repair Tool")
st.markdown("Upload a damaged ID card, mark the broken areas, and we'll repair it automatically.")

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
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Resize if too large (to avoid performance issues)
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

    # Save the image to a temporary file (required for st_canvas)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_path = tmp_file.name
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(tmp_path, format="PNG")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✏️ Mark Damage")
        st.markdown("Use the brush to paint over damaged areas.")
        # Use the temporary file path as background_image
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",   # White mask
            stroke_width=brush_size,
            stroke_color="rgba(255, 255, 255, 1)",
            background_image=tmp_path,             # file path (string)
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="freedraw",
            key="canvas",
        )

    # Repair button
    if st.button("🛠️ Repair", type="primary"):
        if canvas_result is not None and canvas_result.image_data is not None:
            # Extract the mask from the alpha channel
            mask_data = canvas_result.image_data[:, :, 3].astype(np.uint8)
            mask = (mask_data > 0).astype(np.uint8) * 255

            # Apply inpainting
            if repair_mode == "Telea":
                inpaint_method = cv2.INPAINT_TELEA
            else:
                inpaint_method = cv2.INPAINT_NS

            repaired = cv2.inpaint(image, mask, 3, inpaint_method)
            repaired_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("✅ Repaired Card")
                st.image(repaired_rgb, use_container_width=True)

                # Download button
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
        else:
            st.warning("Please draw on the image to mark damaged areas.")

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except:
        pass
else:
    st.info("👈 Upload an image to get started.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OpenCV, and streamlit-drawable-canvas.")
