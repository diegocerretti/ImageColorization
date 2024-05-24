"""
models.py

This module provides functions and classes of the models.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from pathlib import Path
from typing import Optional

class CNN(nn.Module):
    """
    A deep Convolutional Neural Network (CNN) model.

    This model consists of nine of convolutional layers, batch normalization,
    and ReLU activations followed by two fully connected layers. The final output
    is reshaped to the expected dimensions.

    Attributes:
        height (int): Height of the input image.
        width (int): Width of the input image.
        features (nn.Sequential): Sequential container of convolutional layers, batch normalization,
                                  and ReLU activations.
        flatten (nn.Flatten): Layer to flatten the output of the convolutional layers.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer that outputs the final prediction.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """

    def __init__(self, height: int, width: int):
        """
        Initializes the CNN model with specified height and width of the input image.

        Args:
            height (int): Height of the input image.
            width (int): Width of the input image.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=2),  # Conv1
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=2),  # Conv2
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2),  # Conv3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),  # Conv4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),  # Conv5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2),  # Conv6
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=2),  # Conv7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=2),  # Conv8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=2, padding=2),  # Conv9
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 2 * self.height * self.width)  # adjust accordingly
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output tensor reshaped to (batch_size, 2, height, width).
        """
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, 2, self.height, self.width)
        return x
    
class UNet(nn.Module):
    """
    A U-Net model for image segmentation tasks.

    This model consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) 
    that enables precise localization. It uses convolutional layers with Leaky ReLU activations and skip connections 
    to combine high-resolution features from the contracting path with the upsampled output.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer in the encoder.
        conv2 (nn.Conv2d): Second convolutional layer in the encoder.
        maxpool1 (nn.MaxPool2d): Max pooling layer after conv2.
        conv4 (nn.Conv2d): Third convolutional layer in the encoder.
        conv5 (nn.Conv2d): Fourth convolutional layer in the encoder.
        maxpool2 (nn.MaxPool2d): Max pooling layer after conv5.
        conv7 (nn.Conv2d): Fifth convolutional layer in the encoder.
        conv8 (nn.Conv2d): Sixth convolutional layer in the encoder.
        maxpool3 (nn.MaxPool2d): Max pooling layer after conv8.
        conv10 (nn.Conv2d): Seventh convolutional layer in the encoder.
        conv11 (nn.Conv2d): Eighth convolutional layer in the encoder.
        maxpool4 (nn.MaxPool2d): Max pooling layer after conv11.
        conv13 (nn.Conv2d): First convolutional layer in the bottleneck.
        conv14 (nn.Conv2d): Second convolutional layer in the bottleneck.
        up1 (nn.ConvTranspose2d): First upsampling layer in the decoder.
        conv16 (nn.Conv2d): Ninth convolutional layer in the decoder.
        conv17 (nn.Conv2d): Tenth convolutional layer in the decoder.
        up2 (nn.ConvTranspose2d): Second upsampling layer in the decoder.
        conv19 (nn.Conv2d): Eleventh convolutional layer in the decoder.
        conv20 (nn.Conv2d): Twelfth convolutional layer in the decoder.
        up3 (nn.ConvTranspose2d): Third upsampling layer in the decoder.
        conv22 (nn.Conv2d): Thirteenth convolutional layer in the decoder.
        conv23 (nn.Conv2d): Fourteenth convolutional layer in the decoder.
        up4 (nn.ConvTranspose2d): Fourth upsampling layer in the decoder.
        conv25 (nn.Conv2d): Fifteenth convolutional layer in the decoder.
        conv26 (nn.Conv2d): Sixteenth convolutional layer in the decoder.
        output (nn.Conv2d): Output convolutional layer that produces the final segmentation map.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
    """

    def __init__(self):
        """
        Initializes the U-Net model.
        """
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.conv13 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv14 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)

        self.conv16 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv17 = nn.Conv2d(512, 512, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)

        self.conv19 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv20 = nn.Conv2d(256, 256, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)

        self.conv22 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv23 = nn.Conv2d(128, 128, 3, padding=1)
        self.up4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)

        self.conv25 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv26 = nn.Conv2d(64, 64, 3, padding=1)
        self.output = nn.Conv2d(64, 2, 1, padding=0)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2, height, width).
        """
        x1 = leaky_relu(self.conv1(x), negative_slope=0.2)
        x1 = leaky_relu(self.conv2(x1), negative_slope=0.2)
        x = self.maxpool1(x1)

        x2 = leaky_relu(self.conv4(x), negative_slope=0.2)
        x2 = leaky_relu(self.conv5(x2), negative_slope=0.2)
        x = self.maxpool2(x2)

        x3 = leaky_relu(self.conv7(x), negative_slope=0.2)
        x3 = leaky_relu(self.conv8(x3), negative_slope=0.2)
        x = self.maxpool3(x3)

        x4 = leaky_relu(self.conv10(x), negative_slope=0.2)
        x4 = leaky_relu(self.conv11(x4), negative_slope=0.2)
        x = self.maxpool4(x4)

        x = leaky_relu(self.conv13(x), negative_slope=0.2)
        x = leaky_relu(self.conv14(x), negative_slope=0.2)
        x = self.up1(x)

        x = torch.cat([x4, x], dim=1)
        x = leaky_relu(self.conv16(x), negative_slope=0.2)
        x = leaky_relu(self.conv17(x), negative_slope=0.2)
        x = self.up2(x)

        x = torch.cat([x3, x], dim=1)
        x = leaky_relu(self.conv19(x), negative_slope=0.2)
        x = leaky_relu(self.conv20(x), negative_slope=0.2)
        x = self.up3(x)

        x = torch.cat([x2, x], dim=1)
        x = leaky_relu(self.conv22(x), negative_slope=0.2)
        x = leaky_relu(self.conv23(x), negative_slope=0.2)
        x = self.up4(x)

        x = torch.cat([x1, x], dim=1)
        x = leaky_relu(self.conv25(x), negative_slope=0.2)
        x = leaky_relu(self.conv26(x), negative_slope=0.2)
        x = self.output(x)

        return torch.sigmoid(x)

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

    def forward(self, x: torch.Tensor):
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

def save_model(model: torch.nn.Module, model_name: str, model_dir: Optional[str] = "models"):
    """
    Save a PyTorch model to a pth file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name of the model file (e.g., "baseline"). No extension needed.
        model_dir (Optional[str]): The directory where the model file will be saved. Default is "models".
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

