import cv2
def preprocess_image(image):
    """
    Performs basic preprocessing for the
    Digital Eye Strain Predictor.
    """
    if image is None:
        return None

    resized = cv2.resize(image, (512, 512))
    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )
    hsv = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2HSV
    )
    blur = cv2.GaussianBlur(
        resized,
        (5, 5),
        0
    )
    return {
        "resized": resized,
        "rgb": rgb,
        "gray": gray,
        "hsv": hsv,
        "blur": blur
    }