import os
import json
import cv2
import numpy as np
import tensorflow as tf
import threading
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
import webview
# Load class names from JSON file
def load_class_names(file_path='class_names.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load class names
class_names = load_class_names()

# Load the model
model_path = "traffic_sign_classifier.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file at path '{model_path}' not found.")
model = tf.keras.models.load_model(model_path)

# Preprocess image: resize, normalize, and add batch dimension
def preprocess_image(image_path, img_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found or cannot be opened.")
    img_resized = cv2.resize(img, img_size)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    return img_array

# Resize and normalize example images
def resize_and_normalize_image(image_path, save_path, img_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Image at path '{image_path}' not found or cannot be opened.")
        return False
    img_resized = cv2.resize(img, img_size)
    cv2.imwrite(save_path, img_resized)
    return True

# Resize the uploaded image
def resize_uploaded_image(file, target_size=(256, 256)):
    file_content = file.read()
    img = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("Uploaded image cannot be opened.")
    img_resized = cv2.resize(img, target_size)
    return img_resized

# Classify traffic sign image
def classify_traffic_sign(image_path, img_size=(64, 64)):
    img_array = preprocess_image(image_path, img_size)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return class_names[class_idx], confidence

# Flask app setup
app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Ensure the example images directory exists
example_images_dir = 'static/example_images/'
if not os.path.exists(example_images_dir):
    os.makedirs(example_images_dir)

# Resize and normalize example images
for class_name in class_names:
    original_path = os.path.join(example_images_dir, f"{class_name}.png")
    resized_path = os.path.join(example_images_dir, f"resized_{class_name}.png")
    resize_and_normalize_image(original_path, resized_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Index route for the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        return render_template('result.html')

# Route to handle the uploaded image and classification
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    try:
        resized_img = resize_uploaded_image(file)
        file_path = os.path.join("uploads", file.filename)
        cv2.imwrite(file_path, resized_img)
    except Exception as e:
        return jsonify(error=f"Error processing uploaded image: {str(e)}"), 500
    try:
        traffic_sign_type, confidence = classify_traffic_sign(file_path)
        example_image_url = url_for('static', filename=f'example_images/resized_{traffic_sign_type}.png')
        uploaded_image_url = url_for('uploaded_file', filename=file.filename)
        confidence = float(confidence)
        return jsonify(
            traffic_sign_type=traffic_sign_type,
            confidence=confidence,
            uploaded_image_url=uploaded_image_url,
            example_image_url=example_image_url
        )
    except Exception as e:
        return jsonify(error=f"Error during classification: {str(e)}"), 500

def start_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()
    webview.create_window("Traffic Sign Classifier", "http://127.0.0.1:5000/")
    webview.start()
