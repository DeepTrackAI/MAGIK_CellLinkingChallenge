import os
import glob

import torch
from torch import nn

import cv2

import deeplay as dl


_DATASET_PATH = {
    "DIC-C2DH-HeLa": "logs/DIC-C2DH-HeLa/checkpoint.ckpt",
    "BF-C2DL-MuSC": "logs/BF-C2DL-MuSC/checkpoint.ckpt",
    "Fluo-C2DL-MSC": "logs/Fluo-C2DL-MSC/checkpoint.ckpt",
    "Fluo-N2DH-GOWT1": "logs/Fluo-N2DH-GOWT1/checkpoint.ckpt",
    "Fluo-N2DH-SIM+": "logs/Fluo-N2DH-SIM+/checkpoint.ckpt",
    "PhC-C2DH-U373": "logs/PhC-C2DH-U373/checkpoint.ckpt",
    "PhC-C2DL-PSC": "logs/PhC-C2DL-PSC/checkpoint.ckpt",
    "Fluo-N2DL-HeLa": "logs/Fluo-N2DL-HeLa/checkpoint.ckpt",
    "BF-C2DL-HSC": "logs/BF-C2DL-HSC/checkpoint.ckpt",
}


def count_numeric_characters(s):
    """
    Count the number of numeric characters in a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    int
        Number of numeric characters.
    """
    # Handle platform-specific path delimiters
    slast = os.path.basename(s)
    return sum(c.isdigit() for c in slast)


def load_images(path):
    """
    Load a list of images from a directory.

    Parameters
    ----------
    path : str
        Path to the directory containing the images.

    Returns
    -------
    list of numpy.ndarray
        List of images.
    """
    files = sorted(glob.glob(os.path.join(path, "*.tif")))
    return [
        cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in files
    ], count_numeric_characters(files[0])


def load_model(dataset_name):
    """
    Load a pre-trained model for a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    torch.nn.Module
        Pre-trained model.
    """
    if dataset_name not in _DATASET_PATH:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available datasets: {list(_DATASET_PATH.keys())}"
        )
    path = _DATASET_PATH[dataset_name]

    model = dl.GraphToEdgeMAGIK(
        [
            96,
        ]
        * 4,
        1,
        out_activation=nn.Sigmoid,
    )
    classifier = dl.BinaryClassifier(model).create()
    # classifier.load_state_dict(torch.load(path)["state_dict"])
    classifier.load_state_dict(torch.load(path, map_location=torch.device('cpu'))["state_dict"])
    # classifier.to("cuda" if torch.cuda.is_available() else "cpu")

    return classifier
