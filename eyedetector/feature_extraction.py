import cv2
import numpy as np
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    blue_channel = image[:, :, 0]
    blue_light = np.mean(blue_channel)
    saturation = np.mean(hsv[:, :, 1])
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = (edge_pixels / total_pixels) * 100

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Blue Light Intensity": blue_light,
        "Saturation": saturation,
        "Sharpness": sharpness,
        "Visual Clutter": edge_density
    }
    return features