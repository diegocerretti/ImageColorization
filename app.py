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
import matplotlib.pyplot as plt
import io
import datetime
import torch
import torchvision.transforms as transforms
from utils.models import UNet, EncoderDecoderGenerator, load_model
from utils.plots import plot_model_pred
from skimage.color import rgb2lab

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="Image Colorization", page_icon=":palette:")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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
(CNN), a U-Net, and a generative adversarial network (GAN). For the CNN and U-Net, we use\
three loss functions to understand their impact on the colorization properties. We evaluate the\
models’ performances using mean squared error (MSE), peak signal-to-noise ratio (PSNR), structural\
similarity index measure (SSIM), and Fréchet inception distance (FID) score. The results indicate\
that CNNs struggle to capture the color structure of images, whereas U-Nets achieve significantly\
better colorization across all evaluation metrics. GANs, although challenging to train, demonstrate\
comparable performance to U-Nets and show potential for improvement with additional tuning.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/17.png", caption="Grayscale Elvis' Image", use_column_width=True)
    with col2:
        st.image("outputs/18.png", caption="Colorized Elvis' Image", use_column_width=True)

    st.markdown("## Useful Links")
    st.write("- [GitHub Repository](https://github.com/sandromikautadze/image-colorization)")
    st.write("- [Research Report (TBA)](https://www.google.com/)")

def colorize_page():
    st.markdown("# Colorize Your Images")
    st.write("Try two of our models to colorize your own images!")
    st.write("Please notice that **the results will be cropped to a square resolution (1:1) and downsized to 256x256**.\
    For this reason we **strongly** suggest cropping the uploaded images either beforehand or with the cropping tool of the webpage.")
    uploaded_files = st.file_uploader("**Upload Images**", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = load_image(uploaded_file)
            st.markdown("### Original Image")
            container = st.container()
            col1, col2, col3 = container.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Original Image", use_column_width=False, width=500)
            crop_option = st.radio("**Do you want to crop the image to a square (1:1) resolution?**", ["No", "Yes"])
            if crop_option == "Yes":
                st.markdown("### Select Crop Area")
                size_pct = st.slider("Image Crop (%)",
                                     min_value=0.01, max_value=1.0, value=0.5, step=0.01, format="%.2f",
                                     help="Adjust the slider to set the crop size as a percentage of the original image size. The higher the value the bigger the image")
                left_pct = st.slider("Left Crop (%)",
                                     min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
                                     help="Adjust the slider to set the left position of the crop area. The higher the value, the further right")
                top_pct = st.slider("Bottom Crop (%)",
                                    min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
                                    help="Adjust the slider to set the top position of the crop area. The higher the value, the further below)")
                cropped_image = crop_image(image, left_pct, top_pct, size_pct)
                st.markdown("### Cropped Image")
                crop_container = st.container()
                col1, col2, col3 = crop_container.columns([1, 2, 1])
                with col2:
                    st.image(cropped_image, caption="Cropped Image", use_column_width=False, width=500)
            else:
                cropped_image = image
                
            selected_model = st.selectbox("Select a Model", list(models.keys()), index=0)
            model = models[selected_model]

            if st.button("Colorize"):
                with st.spinner("Colorizing image..."):
                    lab = rgb2lab(transform(cropped_image).permute(1, 2, 0).numpy())
                    l_channel = torch.from_numpy(lab[:, :, 0] / 100).unsqueeze(0).to(device)

                    st.markdown("### Grayscale and Colorized Images")

                    colorized_image = plot_model_pred(l_channel, model, device=device)
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    buf.seek(0)
                    colorized_pil = Image.open(buf)

                    # Display grayscale and colorized images side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cropped_image.convert("L"), caption="Grayscale Image", use_column_width=True)
                    with col2:
                        st.image(colorized_pil, caption="Colorized Image", use_column_width=True)

                    buf.close()

                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
                    filename = f"colorized_image_{timestamp}.png"
                    colorized_pil.save(filename)
                    buf = io.BytesIO()
                    colorized_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(label="Download Colorized Image", data=byte_im, file_name=filename)
                    
# def how():
    # st.markdown("# How Does Colorization Work? 🎨")
    # st.write("To be added.")
    # st.write("Ah, I see you're a brave soul willing to dive into the depths of colorization! Buckle up, my friend, for a wild ride through the world of AI and image manipulation!")

    # st.markdown("## The Color Space Conundrum")
    # st.write("Before we delve into the colorization magic, we must first understand the realm of color spaces. You see, computers are excellent at dealing with numbers, but colors? Well, that's a whole different ballgame.")
    # st.write("Enter the **Lab color space**, our trusty sidekick in the colorization adventure. This clever color model separates the luminance (brightness) from the chrominance (color) information, making it easier for our AI models to work their magic.\
    #     In the Lab space, the 'L' channel represents the luminance, while the 'a' and 'b' channels encode the green-red and blue-yellow color components, respectively. By feeding our models the grayscale 'L' channel, they can predict the 'a' and 'b' channels, effectively colorizing the image!")

    # st.markdown("## The Models")
    # st.markdown("### The U-Net")
    # st.write("Now, let's meet our first colorization champion: the U-Net! This architecture, originally designed for biomedical image segmentation, has proven to be a real MVP in the world of colorization.")
    # st.write("The U-Net is a fully convolutional neural network (CNN) with a unique U-shaped structure. It starts with an encoder path that downsamples the input image, capturing higher-level features. Then, it transitions into a decoder path that upsamples the encoded information, reconstructing the image to its original size.")
    # st.write("But here's the real kicker: the U-Net has skip connections that allow the decoder to reuse and combine the low-level features from the encoder. This helps it preserve crucial details and produce more accurate colorizations.")

    # st.markdown("### The GAN")
    # st.write("Next up, we have the Generative Adversarial Network (GAN), a dynamic duo of a generator and a discriminator locked in an eternal dance of deception.")
    # st.write("The generator, our colorful artist, takes the grayscale image and tries to produce a realistic colorization that can fool the discriminator. Meanwhile, the discriminator, our sharp-eyed critic, examines the colorized image and real color images, trying to distinguish the fakes from the genuine ones.")
    # st.write("As the training progresses, the generator becomes better at creating convincing colorizations, while the discriminator becomes more adept at spotting the fakes. This back-and-forth battle ultimately leads to highly realistic and vibrant colorizations (or so we hope!).")

    # st.markdown("## The Colorization Process 🔄")
    # st.write("Alright, now that you've met the key players, let's dive into the colorization process itself!")
    # st.write("1. First, we feed our grayscale image (the 'L' channel) into the chosen model (U-Net or GAN).")
    # st.write("2. The model works its magic, using its intricate network of layers and connections to predict the 'a' and 'b' color channels.")
    # st.write("3. Once the model has produced its colorized output, we combine the predicted 'a' and 'b' channels with the original 'L' channel to reconstruct the full-color image.")
    # st.write("4. And voilà! Your once-dull grayscale image is now a vibrant, colorful masterpiece, ready to be admired and shared with the world!")

    # st.write("Of course, this is just a simplified explanation, and the actual process involves complex mathematical operations, loss functions, and optimization techniques that would make even the most seasoned mathematician's head spin. But hey, that's what makes it fun, right?")

    # st.write("So, there you have it! The secrets of colorization, unveiled in all their glory. Now go forth and colorize to your heart's content, you brave explorer of the digital realms!")
    
    # st.markdown("## For the Brave Souls 💥")
    # st.write("But wait, there's more! For those of you who crave even more technical details, buckle up and hold on tight, because we're about to take a deep dive into the inner workings of our colorization models.")
    # st.markdown("### The U-Net Architecture")
    # st.write("Our U-Net is a fully convolutional neural network with a unique U-shaped structure, consisting of an encoder path and a decoder path.")
    # st.write("The encoder path progressively downsamples the input image, capturing higher-level features, while the decoder path upsamples this encoded information, reconstructing the colorized image to its original size.")
    # st.write("But here's the real kicker: the U-Net has skip connections that allow the decoder to reuse and combine the low-level features from the encoder. This helps it preserve crucial details and produce more accurate colorizations.")
    # st.write("To train our U-Net, we used the Adam optimizer and experimented with various loss functions, including mean squared error (MSE), L1 loss, and smooth L1 loss. Each loss function had its own unique way of penalizing the difference between the predicted and ground truth colors, leading to slightly different colorization results.")
    # st.markdown("### The GAN Architecture")
    # st.write("Our GAN (Generative Adversarial Network) is a dynamic duo of a generator and a discriminator, locked in an eternal battle of deception.")
    # st.write("The generator is an encoder-decoder architecture, where the encoder takes the grayscale image and progressively downsamples it, capturing higher-level features. The decoder then upsamples this encoded information, reconstructing the colorized image.")
    # st.write("But wait, there's a twist! We used a PatchGAN discriminator, which means it doesn't just judge the entire image as real or fake; instead, it classifies whether individual patches of the image look realistic or not. This helps the generator focus on producing more locally coherent and high-quality colorizations.")
    # st.write("To train our GAN, we used the Adam optimizer for the generator and Nesterov SGD for the discriminator, with a carefully curated mix of regularization techniques like L1 regularization, label smoothing, and Gaussian noise. It was a delicate dance of hyperparameters and tricks to avoid common GAN pitfalls like mode collapse and vanishing gradients.")
    
    # st.write("Phew, that was a lot of technical jargon, wasn't it? But fear not, dear explorers, for the journey to true understanding is paved with such challenges. Embrace the complexities, and you shall emerge victorious, wielding the power of colorization like a true AI wizard!")

    # st.write("So, there you have it! The secrets of colorization, unveiled in all their glory (and technical complexity). Now go forth and colorize to your heart's content, you brave explorer of the digital realms!")

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    pages = ["Home Page", "Colorize Your Images"]
    page = st.sidebar.radio("Go to", pages)

    if page == "Home Page":
        main_page()
    elif page == "Colorize Your Images":
        colorize_page()
    # elif page == "How Does Colorization Work?":
        # how()

    # Move the credit text to the bottom of the sidebar
    st.sidebar.markdown("<p style='text-align: center; font-size: 10px; color: gray;'>Webapp generated by Sandro Mikautadze</p>", unsafe_allow_html=True)
