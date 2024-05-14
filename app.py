"""
app.py

This script creates a Streamlit web application for image colorization using a pre-trained PyTorch model.
The application allows users to upload grayscale or color images, which are then colorized using the
trained model. The original and colorized images are displayed side by side for comparison.

Run using the command "streamlit run app.py"

TO BE COMPLETED ONCE THE FINAL MODELS ARE ALL TRAINED

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils.models import ColorizationCNN, load_model
from utils.plots import plot_l, plot_rgb, reconstruct_lab

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model = load_model(ColorizationCNN(), "models/baseline.pth").to(device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def main():
    st.title("Image Colorization App")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        rgb_image = transform(image).to(device)
        image_grayscale = image.convert("L")
        l_channel = transform(image_grayscale).unsqueeze(0).to(device)

        # Colorize the image
        with torch.no_grad():
            ab_channels = loaded_model(l_channel)

        # Display the results
        st.header("Original RGB Image")
        st.image(image, use_column_width=True)
        plot_rgb(rgb_image.cpu())

        st.header("Original Grayscale Image")
        st.image(image_grayscale, use_column_width=True)
        plot_l(l_channel.cpu().squeeze(0))

        st.header("Colorized Image")
        reconstruct_lab(l_channel.cpu(), (ab_channels.cpu()[0], ab_channels.cpu()[1]))

if __name__ == "__main__":
    main()
