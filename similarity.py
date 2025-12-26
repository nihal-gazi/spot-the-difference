import cv2
import numpy as np

def pooled_descriptor(img):
    img = cv2.resize(img, (128, 128))
    return img.flatten()

def similarity(a, b):
    return np.linalg.norm(a - b)
