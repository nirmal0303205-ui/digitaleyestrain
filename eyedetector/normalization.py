def normalize(value, minimum, maximum):
    """
    Converts a value into a 0-100 scale.
    """

    # Prevent division by zero
    if maximum == minimum:
        return 50

    normalized = ((value - minimum) / (maximum - minimum)) * 100

    # Keep the result between 0 and 100
    normalized = max(0, min(100, normalized))

    return normalized


def normalize_features(features):

    normalized_features = {}

    # Project-defined ranges
    normalized_features["Brightness"] = normalize(
        features["Brightness"], 0, 255
    )

    normalized_features["Contrast"] = normalize(
        features["Contrast"], 0, 128
    )

    normalized_features["Blue Light Intensity"] = normalize(
        features["Blue Light Intensity"], 0, 255
    )

    normalized_features["Saturation"] = normalize(
        features["Saturation"], 0, 255
    )

    normalized_features["Sharpness"] = normalize(
        features["Sharpness"], 0, 1000
    )

    normalized_features["Visual Clutter"] = normalize(
        features["Visual Clutter"], 0, 100
    )

    return normalized_features