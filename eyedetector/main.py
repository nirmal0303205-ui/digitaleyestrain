import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from preprocessing import preprocess_image
from feature_extraction import extract_features
from normalization import normalize_features
from scoring import calculate_eye_comfort_score

class DigitalEyeStrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Eye Strain Predictor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.image = None
        self.image_path = None
        self.photo = None
        self.chart_canvas = None
        title = tk.Label(
            root,
            text="Digital Eye Strain Predictor",
            font=("Arial", 26, "bold")
        )

        title.pack(pady=(15, 5))

        subtitle = tk.Label(
            root,
            text="Movie Color Grading - Image Analysis System",
            font=("Arial", 13)
        )

        subtitle.pack(pady=(0, 15))

        # ------------------------------------------------
        # BUTTONS
        # ------------------------------------------------

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        upload_button = tk.Button(
            button_frame,
            text="Upload Movie Frame",
            font=("Arial", 12, "bold"),
            width=20,
            command=self.upload_image
        )

        upload_button.grid(
            row=0,
            column=0,
            padx=10
        )

        analyze_button = tk.Button(
            button_frame,
            text="Analyze Image",
            font=("Arial", 12, "bold"),
            width=20,
            command=self.analyze_image
        )

        analyze_button.grid(
            row=0,
            column=1,
            padx=10
        )

        reset_button = tk.Button(
            button_frame,
            text="Reset",
            font=("Arial", 12),
            width=15,
            command=self.reset
        )

        reset_button.grid(
            row=0,
            column=2,
            padx=10
        )
        main_frame = tk.Frame(root)
        main_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        # LEFT SIDE
        left_frame = tk.Frame(
            main_frame,
            bd=2,
            relief="groove"
        )

        left_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        # RIGHT SIDE
        right_frame = tk.Frame(
            main_frame,
            bd=2,
            relief="groove"
        )

        right_frame.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        # Make columns expandable
        main_frame.columnconfigure(
            0,
            weight=1
        )

        main_frame.columnconfigure(
            1,
            weight=1
        )

        main_frame.rowconfigure(
            0,
            weight=1
        )


        image_title = tk.Label(
            left_frame,
            text="Selected Movie Frame",
            font=("Arial", 16, "bold")
        )

        image_title.pack(pady=10)

        self.image_label = tk.Label(
            left_frame,
            text="No image selected",
            width=50,
            height=12
        )

        self.image_label.pack(
            padx=10,
            pady=5
        )

        # ------------------------------------------------
        # RESULTS
        # ------------------------------------------------

        result_title = tk.Label(
            left_frame,
            text="Feature Analysis",
            font=("Arial", 16, "bold")
        )

        result_title.pack(pady=(15, 5))

        self.result_label = tk.Label(
            left_frame,
            text="Upload an image and click Analyze Image",
            font=("Arial", 11),
            justify="left"
        )

        self.result_label.pack(
            padx=10,
            pady=5
        )

        # ------------------------------------------------
        # SCORE
        # ------------------------------------------------

        self.score_label = tk.Label(
            left_frame,
            text="Eye Comfort Score: -- / 100",
            font=("Arial", 18, "bold")
        )

        self.score_label.pack(
            pady=(15, 5)
        )

        self.classification_label = tk.Label(
            left_frame,
            text="Classification: --",
            font=("Arial", 15, "bold")
        )

        self.classification_label.pack(
            pady=5
        )

        # ------------------------------------------------
        # RECOMMENDATIONS
        # ------------------------------------------------

        recommendation_title = tk.Label(
            left_frame,
            text="Recommendations",
            font=("Arial", 15, "bold")
        )

        recommendation_title.pack(
            pady=(15, 5)
        )

        self.recommendation_label = tk.Label(
            left_frame,
            text="Recommendations will appear here.",
            font=("Arial", 10),
            justify="left",
            wraplength=500
        )

        self.recommendation_label.pack(
            padx=15,
            pady=5
        )

        # =================================================
        # RIGHT SIDE - CHART
        # =================================================

        chart_title = tk.Label(
            right_frame,
            text="Feature Analysis Chart",
            font=("Arial", 18, "bold")
        )

        chart_title.pack(
            pady=15
        )

        chart_description = tk.Label(
            right_frame,
            text="Normalized feature values (0 - 100)",
            font=("Arial", 11)
        )

        chart_description.pack(
            pady=(0, 5)
        )

        # Frame where matplotlib chart will appear
        self.chart_frame = tk.Frame(
            right_frame
        )

        self.chart_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Initial message
        self.chart_message = tk.Label(
            self.chart_frame,
            text="Chart will appear after image analysis.",
            font=("Arial", 12)
        )

        self.chart_message.pack(
            expand=True
        )


    # ====================================================
    # UPLOAD IMAGE
    # ====================================================

    def upload_image(self):

        file_path = filedialog.askopenfilename(
            title="Select Movie Frame",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.image_path = file_path

        # Read image
        self.image = cv2.imread(
            file_path
        )

        if self.image is None:

            messagebox.showerror(
                "Error",
                "Unable to load the selected image."
            )

            return

        # Convert BGR → RGB
        rgb_image = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2RGB
        )

        # Convert to PIL
        pil_image = Image.fromarray(
            rgb_image
        )

        # Resize image for display
        pil_image.thumbnail(
            (500, 300)
        )

        # Convert to Tkinter image
        self.photo = ImageTk.PhotoImage(
            pil_image
        )

        # Display
        self.image_label.config(
            image=self.photo,
            text=""
        )

        # Reset results
        self.result_label.config(
            text="Image uploaded successfully.\nClick Analyze Image."
        )

        self.score_label.config(
            text="Eye Comfort Score: -- / 100"
        )

        self.classification_label.config(
            text="Classification: --"
        )

        self.recommendation_label.config(
            text="Recommendations will appear here."
        )

        self.clear_chart()

        self.chart_message = tk.Label(
            self.chart_frame,
            text="Chart will appear after image analysis.",
            font=("Arial", 12)
        )

        self.chart_message.pack(
            expand=True
        )


    # ====================================================
    # ANALYZE IMAGE
    # ====================================================

    def analyze_image(self):

        if self.image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload an image first."
            )

            return

        try:

            # ============================================
            # STEP 1 - PREPROCESSING
            # ============================================

            processed = preprocess_image(
                self.image
            )

            if processed is None:

                messagebox.showerror(
                    "Error",
                    "Image preprocessing failed."
                )

                return

            processed_image = processed["resized"]


            # ============================================
            # STEP 2 - FEATURE EXTRACTION
            # ============================================

            features = extract_features(
                processed_image
            )


            # ============================================
            # STEP 3 - NORMALIZATION
            # ============================================

            normalized_features = normalize_features(
                features
            )


            # ============================================
            # STEP 4 - EYE COMFORT SCORE
            # ============================================

            score = calculate_eye_comfort_score(
                normalized_features
            )


            # ============================================
            # STEP 5 - CLASSIFICATION
            # ============================================

            if score >= 70:

                classification = "Comfortable"

            elif score >= 40:

                classification = "Moderate"

            else:

                classification = "Eye-Straining"


            # ============================================
            # STEP 6 - DISPLAY FEATURES
            # ============================================

            result_text = (
                f"Brightness       : {features['Brightness']:.2f}\n"
                f"Contrast         : {features['Contrast']:.2f}\n"
                f"Blue Light       : {features['Blue Light Intensity']:.2f}\n"
                f"Saturation       : {features['Saturation']:.2f}\n"
                f"Sharpness        : {features['Sharpness']:.2f}\n"
                f"Visual Clutter   : {features['Visual Clutter']:.2f}"
            )

            self.result_label.config(
                text=result_text
            )


            # ============================================
            # STEP 7 - DISPLAY SCORE
            # ============================================

            self.score_label.config(
                text=f"Eye Comfort Score: {score:.2f} / 100"
            )


            self.classification_label.config(
                text=f"Classification: {classification}"
            )


            # ============================================
            # STEP 8 - RECOMMENDATIONS
            # ============================================

            recommendations = self.generate_recommendations(
                normalized_features,
                score
            )

            self.recommendation_label.config(
                text=recommendations
            )


            # ============================================
            # STEP 9 - CREATE CHART
            # ============================================

            self.create_chart(
                normalized_features
            )


        except Exception as error:

            messagebox.showerror(
                "Analysis Error",
                str(error)
            )


    # ====================================================
    # RECOMMENDATIONS
    # ====================================================

    def generate_recommendations(
        self,
        features,
        score
    ):

        recommendations = []

        if features["Brightness"] > 70:

            recommendations.append(
                "• Brightness is relatively high."
            )

        elif features["Brightness"] < 20:

            recommendations.append(
                "• Brightness is relatively low."
            )


        if features["Contrast"] > 70:

            recommendations.append(
                "• High contrast detected."
            )


        if features["Blue Light Intensity"] > 70:

            recommendations.append(
                "• High blue-channel intensity detected."
            )


        if features["Saturation"] > 70:

            recommendations.append(
                "• High color saturation detected."
            )


        if features["Sharpness"] > 70:

            recommendations.append(
                "• High sharpness detected."
            )


        if features["Visual Clutter"] > 70:

            recommendations.append(
                "• High visual clutter detected."
            )


        if score < 40:

            recommendations.append(
                "• Overall score indicates high visual intensity."
            )

        elif score < 70:

            recommendations.append(
                "• Moderate visual intensity detected."
            )

        else:

            recommendations.append(
                "• Overall visual characteristics appear relatively comfortable."
            )


        if not recommendations:

            return "No specific recommendations."

        return "\n".join(
            recommendations
        )


    # ====================================================
    # CREATE CHART
    # ====================================================

    def create_chart(
        self,
        normalized_features
    ):

        # Remove previous chart
        self.clear_chart()

        # -----------------------------------------------
        # Feature names
        # -----------------------------------------------

        feature_names = [
            "Brightness",
            "Contrast",
            "Blue Light",
            "Saturation",
            "Sharpness",
            "Visual Clutter"
        ]

        # -----------------------------------------------
        # Normalized values
        # -----------------------------------------------

        values = [
            normalized_features["Brightness"],
            normalized_features["Contrast"],
            normalized_features["Blue Light Intensity"],
            normalized_features["Saturation"],
            normalized_features["Sharpness"],
            normalized_features["Visual Clutter"]
        ]

        # -----------------------------------------------
        # Create matplotlib figure
        # -----------------------------------------------

        figure = plt.Figure(
            figsize=(6, 5),
            dpi=100
        )

        axis = figure.add_subplot(111)

        # -----------------------------------------------
        # Create bars
        # -----------------------------------------------

        bars = axis.bar(
            feature_names,
            values
        )

        # -----------------------------------------------
        # Chart settings
        # -----------------------------------------------

        axis.set_title(
            "Normalized Feature Analysis",
            fontsize=14,
            fontweight="bold"
        )

        axis.set_ylabel(
            "Intensity (0 - 100)"
        )

        axis.set_ylim(
            0,
            100
        )

        # Rotate feature names
        axis.tick_params(
            axis="x",
            rotation=35
        )

        # Grid
        axis.grid(
            axis="y",
            alpha=0.3
        )

        # -----------------------------------------------
        # Display values above bars
        # -----------------------------------------------

        for bar, value in zip(
            bars,
            values
        ):

            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2,
                f"{value:.1f}",
                ha="center",
                fontsize=9
            )

        # -----------------------------------------------
        # Fix layout
        # -----------------------------------------------

        figure.tight_layout()

        # -----------------------------------------------
        # Embed chart inside Tkinter
        # -----------------------------------------------

        self.chart_canvas = FigureCanvasTkAgg(
            figure,
            master=self.chart_frame
        )

        self.chart_canvas.draw()

        canvas_widget = (
            self.chart_canvas.get_tk_widget()
        )

        canvas_widget.pack(
            fill="both",
            expand=True
        )


    # ====================================================
    # CLEAR CHART
    # ====================================================

    def clear_chart(self):

        if self.chart_canvas is not None:

            self.chart_canvas.get_tk_widget().destroy()

            self.chart_canvas = None

        # Remove old message if it exists
        for widget in self.chart_frame.winfo_children():

            widget.destroy()


    # ====================================================
    # RESET
    # ====================================================

    def reset(self):

        self.image = None
        self.image_path = None
        self.photo = None

        self.image_label.config(
            image="",
            text="No image selected"
        )

        self.result_label.config(
            text="Upload an image and click Analyze Image"
        )

        self.score_label.config(
            text="Eye Comfort Score: -- / 100"
        )

        self.classification_label.config(
            text="Classification: --"
        )

        self.recommendation_label.config(
            text="Recommendations will appear here."
        )

        self.clear_chart()

        message = tk.Label(
            self.chart_frame,
            text="Chart will appear after image analysis.",
            font=("Arial", 12)
        )

        message.pack(
            expand=True
        )


# ========================================================
# START APPLICATION
# ========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DigitalEyeStrainApp(
        root
    )

    root.mainloop()