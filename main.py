from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Ensure the uploads directory exists
upload_dir = "./uploads"
os.makedirs(upload_dir, exist_ok=True)

# Create directory for ELA images
ela_dir = "./ELA"
os.makedirs(ela_dir, exist_ok=True)

# Load TFLite model
tflite_model_path = "./model/ela_model7_resnet18_20epochs.tflite"

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # Get expected image size

# Feature Engineering - Error Level Analysis (ELA)
def ELA(img_path):
    temp_path = "./ELA/temp.jpg"
    scale_factor = 10
    original = Image.open(img_path).convert("RGB")
    original.save(temp_path, quality=90)
    temporary = Image.open(temp_path)

    # Compute ELA difference
    ela_image = ImageChops.difference(original, temporary)

    # Enhance Contrast
    enhancer = ImageEnhance.Contrast(ela_image)
    ela_image = enhancer.enhance(scale_factor)

    ela_path = os.path.join(ela_dir, "ela_img.jpg")
    ela_image.save(ela_path)
    os.remove(temp_path)
    return ela_path

# Route endpoint to predict the image
@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    ela_path = ELA(file_path)

    try:
        # Load and preprocess the image for TFLite
        img = image.load_img(ela_path, target_size=input_shape)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Set the tensor input
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get the prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])
        tf_pred = prediction[0][0] * 100  # Convert to percentage

        os.remove(file_path)
        os.remove(ela_path)
        return jsonify({'score': "{:.2f}".format(tf_pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
