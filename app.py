
"""
app.py

This script creates a Streamlit web application for image colorization using pre-trained models.
The application allows users to upload grayscale or color images, which are then colorized using the trained model.
The original and colorized images are displayed side by side for comparison.

Run using the command "streamlit run app.py"

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from utils.models import UNet, EncoderDecoderGenerator, load_model
from utils.plots import plot_l, plot_model_pred
from skimage.color import rgb2lab
import io
import datetime

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="Image Colorization App", page_icon=":palette:")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
models = {
    "U-Net": UNet(),
    "GAN": EncoderDecoderGenerator()
}

unet_weights = "models/unet_l1smooth_trained.pth"
gan_weights = "models/localgen0.pth"

models["U-Net"] = load_model(models["U-Net"], unet_weights).to(device)
models["GAN"] = load_model(models["GAN"], gan_weights).to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    return img

def display_image(image, title):
    st.image(image, caption=title, use_column_width=True)

def crop_image(image, left_pct, top_pct, size_pct):
    width, height = image.size
    left = int(left_pct * width)
    top = int(top_pct * height)
    size = int(size_pct * min(width, height))
    right = left + size
    bottom = top + size
    if right > width:
        left = width - size
    if bottom > height:
        top = height - size
    image = image.crop((left, top, left + size, top + size))
    return image

def main_page():
    st.markdown("# Image Colorization with Deep Learning")
    st.write("This is the webpage for the project in the '30562-MACHINE LEARNING AND ARTIFICIAL INTELLIGENCE' course held at Bocconi University in the 2023/2024 academic year.")
    st.write("**Authors**: Beatrice Citterio, Diego Cerretti, Mattia Martino, Sandro Mikautadze")
    st.markdown("## Abstract")
    st.write("In this work, we assess the performance of various deep learning architectures to colorize grayscale\
images, using the MS COCO dataset. We train three main models: a convolutional neural network\
CNN), a U-Net, and a generative adversarial network (GAN). For the CNN and U-Net, we use\
three loss functions to understand their impact on the colorization properties. We evaluate the\
models’ performances using mean squared error (MSE), peak signal-to-noise ratio (PSNR), structural\
similarity index measure (SSIM), and Fréchet inception distance (FID) score. The results indicate\
that CNNs struggle to capture the color structure of images, whereas U-Nets achieve significantly\
better colorization across all evaluation metrics. GANs, although challenging to train, demonstrate\
comparable performance to U-Nets and show potential for improvement with additional tuning.")
    st.markdown("## Useful Links")
    st.write("- [GitHub Repository](https://github.com/sandromikautadze/image-colorization)")
    st.write("- [Research Report (TBA)](https://www.google.com/)")

def colorize_page():
    st.markdown("# Colorize Your Images")
    st.write("Try two of our models to colorize your own images!\
        To do that you need to choose an image and upload it here!")
    st.write("Please notice that **the results will be cropped to a square resolution (1:1) and downsized to 256x256**.\
    For this reason we **strongly** suggest cropping the images either before uploading or with the cropping tool of the webpage.")
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.markdown("### Original Image")
        display_image(image, "Original Image")
        crop_option = st.radio("Do you want to crop the image to a square (1:1) resolution?\nIf 'Yes' the image will not be shrinked; if 'No' the image will be shrinked and distorted", ["No", "Yes"])

        if crop_option == "Yes":
            st.markdown("### Select Crop Area")
            size_pct = st.slider("What fraction of the original image size do you want to keep?", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f", help="Adjust the slider to set the crop size as a percentage of the original image size.")
            left_pct = st.slider("How far right do you want to go? (The higher the value, the farther right)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f", help="Adjust the slider to set the left position of the crop area.")
            top_pct = st.slider("How far below do you want to go? (The higher the value, the farther below)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f", help="Adjust the slider to set the top position of the crop area.")

            cropped_image = crop_image(image, left_pct, top_pct, size_pct)
            st.markdown("### Cropped Image")
            display_image(cropped_image, "Cropped Image")
        else:
            cropped_image = image

        grayscale_image = cropped_image.convert("L")

        st.markdown("---")

        st.markdown("### Select a Model")
        selected_model = st.selectbox("", list(models.keys()), index=0, help="Choose the model to use for colorization.")

        if st.button("Colorize"):
            with st.spinner("Colorizing image..."):
                lab = rgb2lab(transform(cropped_image).permute(1, 2, 0).numpy())
                l_channel = torch.from_numpy(lab[:, :, 0] / 100).unsqueeze(0).to(device)

                st.markdown("### Grayscale Image")
                display_image(cropped_image.convert("L"), "Grayscale Image")

                # now = datetime.datetime.now()
                # timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
                # filename = f"grayscale_image_{timestamp}.png"
                # grayscale_buf = io.BytesIO()
                # grayscale_image.save(grayscale_buf, format="PNG")
                # grayscale_byte_im = grayscale_buf.getvalue()
                # st.download_button(label="Download Grayscale Image", data=grayscale_byte_im, file_name=filename)
                # grayscale_buf.close()

                st.markdown("### Colorized Image")
                model = models[selected_model]
                colorized_image = plot_model_pred(l_channel, model, device=device)
                
                # Convert matplotlib figure to PIL Image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                colorized_pil = Image.open(buf)

                display_image(colorized_pil, "Colorized Image")
                buf.close()

                # Provide download option and save the colorized image with timestamp
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
                filename = f"colorized_image_{timestamp}.png"
                colorized_pil.save(filename)
                buf = io.BytesIO()
                colorized_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="Download Colorized Image", data=byte_im, file_name=filename)

if __name__ == "__main__":
    # Add a sidebar with navigation
    page = st.sidebar.radio("",("Home Page", "Colorize Your Images"))

    if page == "Home Page":
        main_page()
    elif page == "Colorize Your Images":
        colorize_page()

    # Add some vertical spacing
    st.markdown("   ")

    # Add the small text credit
    st.markdown("<p style='text-align: center; font-size: 10px; color: gray;'>Webapp generated by Sandro Mikautadze</p>", unsafe_allow_html=True)