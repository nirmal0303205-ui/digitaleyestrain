import cv2

from feature_extraction import extract_features
from normalization import normalize_features


# Load image
image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found!")

else:
    # Extract raw features
    features = extract_features(image)

    # Normalize features
    normalized = normalize_features(features)

    print("\nNORMALIZED FEATURES")
    print("-------------------")

    for feature, value in normalized.items():
        print(feature, ":", round(value, 2))