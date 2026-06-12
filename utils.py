# utils.py
from PIL import Image
import numpy as np
from config import IMG_SIZE

def preprocess_image(path):
    """
    Load image, convert to RGB, resize to IMG_SIZE, normalize and return (1,H,W,3).
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)
