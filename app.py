import streamlit as st
from PIL import Image

from src.inference import generate_caption


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# ============================================
# Title
# ============================================

st.title("🖼️ Image Caption Generator")

st.write(
    "Upload an image and let the AI generate a natural-language caption."
)


# ============================================
# Image Upload
# ============================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================
# Caption Generation
# ============================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            # Save temporary image
            temp_path = "temp_uploaded_image.jpg"
            image.save(temp_path)

            caption = generate_caption(temp_path)

        st.success("Caption generated successfully!")

        st.subheader("Generated Caption")

        st.write(
            f"**{caption}**"
        )