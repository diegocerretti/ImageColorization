"""
plots.py

This module provides functions for visualizing LAB color channels and RGB images.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb
from typing import Tuple

def plot_l(l_channel: torch.Tensor):
    """
    Plot the L channel of a Lab image.

    Args:
        l_channel (torch.Tensor): Tensor containing the L channel values. Size 1xHxW
    """
    l_channel = l_channel.squeeze()  # Remove unnecessary dimension --> we now have shape [H, W] and not [1, H, W]
    plt.figure(figsize=(6, 6))
    plt.imshow(l_channel, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_a(a_channel: torch.Tensor):
    """
    Plot the A channel of a Lab image.

    Args:
        a_channel (torch.Tensor): Tensor containing the A channel values. Size HxW
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(a_channel, cmap='coolwarm')
    plt.axis('off')
    plt.show()

def plot_b(b_channel: torch.Tensor):
    """
    Plot the B channel of a Lab image.

    Args:
        b_channel (torch.Tensor): Tensor containing the B channel values. Size HxW
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(b_channel, cmap='coolwarm')
    plt.axis('off')
    plt.show()

# FORSE QUESTA FUNZIONE NON CI SERVE A NULLA ALLA FINE
# def plot_ab(ab_channels: torch.Tensor, l_values=[25, 50, 75]):
#     """
#     Plot the AB channels of a Lab image with different L values.

#     Args:
#         ab_channels (torch.Tensor): Tensor containing the AB channel values.
#         l_values (List[int], optional): List of L values to use for plotting. Defaults to [25, 50, 75].
#     """
#     if ab_channels.shape[0] == 2:  # Assuming ab_channels shape is initially (2, H, W)
#         ab_channels = ab_channels.permute(1, 2, 0)  # Change to (H, W, 2)
#     for l in l_values:
#         lab_image = np.zeros((ab_channels.shape[0], ab_channels.shape[1], 3)) # HxWx3
#         lab_image[:, :, 0] = l # Set the L channel to one of the specified L values
#         lab_image[:, :, 1:] = ab_channels * 255 - 128  # Scale AB channels correctly for lab2rgb
#         rgb_image = lab2rgb(lab_image)
#         plt.figure(figsize=(6, 6))
#         plt.imshow(rgb_image)
#         plt.title(f'L = {l}')
#         plt.axis('off')
#         plt.show()

def plot_rgb(rgb_image: torch.Tensor):
    """
    Plot an RGB image.

    Args:
        rgb_image (torch.Tensor): Tensor containing the RGB image values. Size 3xHxW
    """
    rgb_image = rgb_image.permute(1, 2, 0)  # Convert from CxHxW to HxWxC for plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()
    
# We build a function that will later turn useful: given an L channel, and an AB channel, we reconstruct the original image
def reconstruct_lab(l_channel: torch.Tensor, ab_channels: Tuple[torch.Tensor, torch.Tensor]):
    """
    Reconstructs an RGB image from its L, A, and B channels.

    Args:
        l_channel (torch.Tensor): Tensor containing the L channel values. Size 1xHxW. Values in [0,1].
        ab_channels (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the A and B channel tensors. Values in [0,1].
            A channel tensor: Size HxW.
            B channel tensor: Size HxW.
    """
    a_channel = ab_channels[0]
    b_channel = ab_channels[1]
    l_channel = l_channel.squeeze()  # from 1xHxW to HxW
    lab_image = torch.stack((l_channel, a_channel, b_channel), dim=0)

    # Revert normalization of L channel
    l_channel_np = l_channel.cpu().numpy() * 100

    # Revert normalization of A and B channels
    ab_channels_np = torch.stack((a_channel, b_channel), dim=0).permute(1, 2, 0).cpu().numpy() * 255 - 128

    # Stack L, A, B channels
    lab_image_reconstructed = np.zeros((lab_image.shape[1], lab_image.shape[2], 3))
    lab_image_reconstructed[:, :, 0] = l_channel_np
    lab_image_reconstructed[:, :, 1:] = ab_channels_np

    # Convert LAB image to RGB
    rgb_image = lab2rgb(lab_image_reconstructed)

    # Display the RGB image
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()