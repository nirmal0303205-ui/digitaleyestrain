import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
img = None
processed = None

def upload(): 
    global img
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if path:
        img = cv2.imread(path)
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show = Image.fromarray(show)
        show.thumbnail((300, 200))

        photo = ImageTk.PhotoImage(show)
        original.config(image=photo)
        original.image = photo
        status.config(text="Image uploaded successfully")
def preprocess():

    global processed
    if img is None:
        status.config(text="Upload an image first")
        return
    # Resize
    resized = cv2.resize(img, (512, 512))
    # Color conversions
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # Noise reduction
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    processed = [resized, rgb, gray, hsv, blur]

    names = [
        "Resized",
        "RGB",
        "Grayscale",
        "HSV",
        "Noise Reduced"
    ]
    plt.figure(figsize=(10, 6))
    for i, (name, image) in enumerate(zip(names, processed)):
        if name == "Resized" or name == "Noise Reduced":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif name == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        plt.subplot(2, 3, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    status.config(text="Preprocessing completed")
# Feature Analysis
def analyze():
    if img is None:
        status.config(text="Upload an image first")
        return
    image = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate features
    brightness = np.mean(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    saturation = np.mean(hsv[:, :, 1])
    features = [
        "Brightness",
        "Contrast",
        "Sharpness",
        "Saturation"
    ]
    values = [
        brightness,
        contrast,
        sharpness,
        saturation
    ]
    # Print values in terminal
    print("\n--- Feature Values ---")
    for name, value in zip(features, values):
        print(f"{name}: {value:.2f}")
    # Show values in UI
    result.config(
        text=f"""
Brightness : {brightness:.2f}
Contrast   : {contrast:.2f}
Sharpness  : {sharpness:.2f}
Saturation : {saturation:.2f}
"""
    )
    # Feature graph
    plt.figure(figsize=(8, 5))
    plt.bar(features, values)
    plt.title("Image Feature Analysis")
    plt.xlabel("Features")
    plt.ylabel("Raw Value")
    plt.tight_layout()
    plt.show()

root = tk.Tk()
root.title("Digital Eye Strain Predictor")
root.geometry("600x650")
tk.Label(
    root,
    text="Digital Eye Strain Predictor",
    font=("Arial", 20, "bold")
).pack(pady=15)
tk.Button(
    root,
    text="Upload Image",
    command=upload,
    width=20
).pack(pady=5)
original = tk.Label(
    root,
    text="Original Image"
)
original.pack(pady=10)
tk.Button(
    root,
    text="Preprocess Image",
    command=preprocess,
    width=20
).pack(pady=5)
tk.Button(
    root,
    text="Feature Analysis",
    command=analyze,
    width=20
).pack(pady=5)
tk.Label(
    root,
    text="Raw Feature Values",
    font=("Arial", 14, "bold")
).pack(pady=15)
result = tk.Label(
    root,
    text="No analysis yet",
    font=("Arial", 12),
    justify="left"
)
result.pack()
status = tk.Label(
    root,
    text="Upload an image to begin"
)
status.pack(pady=20)
root.mainloop()