"""
models.py

This module provides functions and classes of the models.

Authors: Diego Cerretti, Beatrice Citterio, Mattia Martino, Sandro Mikautadze
"""

import torch
from pathlib import Path

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