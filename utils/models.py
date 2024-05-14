"""
models.py

This module provides functions and classes of the models.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
import torch.nn as nn
from pathlib import Path

class BaselineCNN(nn.Module):
    """
    A baseline CNN for image colorization. Used privately to test functions and other internal things before scaling things up.

    This model takes a grayscale (single-channel) image as input and outputs a two-channel tensor
    representing the predicted A and B channels of the LAB color space. The architecture consists
    of several convolutional layers with ReLU activations, and the final output is passed through a
    sigmoid activation function to ensure the output values are between 0 and 1.

    Attributes:
        layers (nn.Sequential): A sequential container of convolutional layers and ReLU activations.
        sigmoid (nn.Sigmoid): A sigmoid activation function applied to the final output.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Predicted A and B channel tensors of shape (batch_size, 2, height, width).
        """
        x = self.layers(x)
        x = self.sigmoid(x)
        return x

def save_model(model: torch.nn.Module, model_name: str, model_dir: str = "models"):
    """
    Save a PyTorch model to a pth file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name of the model file (e.g., "baseline"). No extension needed.
        model_dir (str, optional): The directory where the model file will be saved. Default is "models".
    """
    # create the model directory if it doesn't exist
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    # construct the full path for the model file
    model_name = f"{model_name}.pth"
    model_path = model_dir_path / model_name

    torch.save(model.state_dict(), model_path) # save the model state dict
    print(f"Model saved to {model_path} successfully!")
    
def load_model(model: torch.nn.Module, model_path: str):
   """
   Load a PyTorch model from a file.

   Args:
       model (torch.nn.Module): The PyTorch model object to load the weights into.
       model_path (str): The path to the model file (e.g., "models/baseline.pth").

   Returns:
       torch.nn.Module: The model object with the loaded weights.
   """
   model.load_state_dict(torch.load(model_path))
   print(f"{model._get_name()} model loaded successfully!")
   return model
