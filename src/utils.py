
import numpy as np
from PIL import Image
import tifffile as tiff

def read_tif(path):
    img = tiff.imread(path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    img = Image.fromarray(img.astype(np.uint8))
    return img
