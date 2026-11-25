import cv2
from feature_extraction import extract_features

# Load image
image = cv2.imread("sample.jpg")

# Check image
if image is None:
    print("Image not found!")
else:

    # Extract features
    features = extract_features(image)

    # Display results
    print("\nIMAGE FEATURE ANALYSIS")
    print("----------------------")

    for feature, value in features.items():
        print(feature, ":", round(value, 2))