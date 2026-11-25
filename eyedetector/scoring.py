def calculate_eye_comfort_score(features):
    """
    Calculate Eye Comfort Score from normalized features.
    All feature values should be between 0 and 100.
    """

    # Feature weights
    brightness_weight = 0.20
    contrast_weight = 0.20
    blue_light_weight = 0.20
    saturation_weight = 0.15
    sharpness_weight = 0.15
    clutter_weight = 0.10

    # Convert "strain" values into "comfort" values.
    # Higher intensity = lower comfort.
    brightness_score = 100 - features["Brightness"]
    contrast_score = 100 - features["Contrast"]
    blue_light_score = 100 - features["Blue Light Intensity"]
    saturation_score = 100 - features["Saturation"]
    sharpness_score = 100 - features["Sharpness"]
    clutter_score = 100 - features["Visual Clutter"]

    # Weighted score
    eye_comfort_score = (
        brightness_score * brightness_weight +
        contrast_score * contrast_weight +
        blue_light_score * blue_light_weight +
        saturation_score * saturation_weight +
        sharpness_score * sharpness_weight +
        clutter_score * clutter_weight
    )

    # Keep score between 0 and 100
    eye_comfort_score = max(0, min(100, eye_comfort_score))

    return eye_comfort_score