import imageio
import numpy as np


im = imageio.imread('images/fox.png').astype(np.float32)
from PIL import Image

im = im.convert('L:')

def srank(A):
    U,S,V=torch.svd(A)
    return torch.sum((S/S.max())**2)

