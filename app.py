import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask import Flask, render_template, request, redirect, url_for
from llama_service import LlamaService

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Llama service
llama_service = LlamaService()

# Use a pre-trained model instead of loading from file
print("Loading pre-trained ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification
classifier = tf.keras.Model(inputs=base_model.input, outputs=output)
print("Model successfully created")

# Define image size
IMG_SIZE = (128, 128)

# Function to refine deforestation mask
def refine_deforestation_mask(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to detect deforested areas
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # Canny edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    return img, mask

# Function to predict deforestation
def predict_deforestation(image_path):
    # Load and preprocess image for ResNet
    img = load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Use ResNet's preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction
    prediction = classifier.predict(img_array)[0][0]
    print(f"Prediction value: {prediction}")

    original, mask = refine_deforestation_mask(image_path)

    # Save original image
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], "original.jpg")
    cv2.imwrite(original_path, original)

    # For demonstration purposes, we'll assume deforestation if prediction > 0.5
    # In a real scenario, you would fine-tune this model on your dataset
    if prediction > 0.5:  # Deforestation detected
        mask_path = os.path.join(app.config["UPLOAD_FOLDER"], "mask.jpg")
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")

        # Create color overlay
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        output = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)

        cv2.imwrite(mask_path, mask)
        cv2.imwrite(output_path, output)

        return True, original_path, mask_path, output_path, prediction

    return False, original_path, None, None, prediction  # No deforestation detected

def get_severity_level(confidence_score):
    """Determine severity level based on confidence score"""
    if confidence_score > 0.8:
        return "severe"
    elif confidence_score > 0.6:
        return "moderate"
    else:
        return "mild"

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
            file.save(file_path)

            deforestation_detected, original, mask, output, confidence = predict_deforestation(
                file_path
            )

            # Get AI-powered insights
            solutions = None
            analytics = None
            prevention_strategies = None
            
            try:
                if deforestation_detected:
                    severity = get_severity_level(confidence)
                    solutions = llama_service.get_deforestation_solutions(
                        location_info="Satellite imagery analysis", 
                        severity=severity
                    )
                
                analytics = llama_service.get_image_analytics(
                    deforestation_detected, 
                    confidence
                )
                
                prevention_strategies = llama_service.get_prevention_strategies()
                
            except Exception as e:
                print(f"Error getting AI insights: {e}")
                # Provide fallback content if API fails
                if deforestation_detected:
                    solutions = "• Implement immediate reforestation programs\n• Establish protected zones\n• Engage local communities in conservation"
                analytics = "• Environmental impact assessment needed\n• Monitor biodiversity changes\n• Track climate implications"
                prevention_strategies = "• Strengthen forest protection policies\n• Implement sustainable land use practices\n• Use satellite monitoring systems"

            return render_template(
                "index.html",
                uploaded_image=original,
                mask=mask,
                output=output,
                result="🛑 Deforestation Detected!" if deforestation_detected else "✅ No Deforestation Detected!",
                confidence=confidence,
                solutions=solutions,
                analytics=analytics,
                prevention_strategies=prevention_strategies,
                deforestation_detected=deforestation_detected
            )

    return render_template("index.html", uploaded_image=None)


if __name__ == "__main__":
    app.run(debug=True)