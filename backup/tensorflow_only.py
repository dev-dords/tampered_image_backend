from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os
import numpy as np
import warnings
# Initialize Flask app

warnings.filterwarnings("ignore");
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Ensure the uploads directory exists
upload_dir = "./uploads"
os.makedirs(upload_dir, exist_ok=True)

# feature engineering - Error Level Analysis
if not os.path.exists('./ELA'):
        os.makedirs('./ELA')

def ELA(img_path):
        dir = f'./ELA/'
        extension = img_path.split('.')[-1]
        temp = "temp." + extension
        scale_factor = 10
        original = Image.open(img_path)
        if(os.path.isdir(dir) == False):
                os.mkdir(dir)
        original.save(temp, quality=90)
        temporary = Image.open(temp)

        #Compute ELA difference
        ela_image = ImageChops.difference(original, temporary)

        #Enhance Contrast
        enhancer = ImageEnhance.Contrast(ela_image)
        ela_image = enhancer.enhance(scale_factor)

        # d = diff.load()
        # WIDTH, HEIGHT = diff.size
        # for x in range(WIDTH):
        #         for y in range(HEIGHT):
        #                 d[x, y] = tuple(k * SCALE for k in d[x, y])
        ela_path = dir + 'ela_img.' + extension
        ela_image.save(ela_path)
        os.remove(temp)
        return ela_path
# Load the model once at startup
# model3= load_model("./model/ela_model3_xception_20epochs.keras")
model4= load_model("./model/ela_model4_xception_32epochs.keras")
model5= load_model("./model/ela_model5_xception_attentnion_18epochs.keras")
model7= load_model("./model/ela_model7_resnet18_20epochs.keras")

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
        # Load and preprocess the image
        # Make the prediction
        predictions = []
        for model in [model4, model5, model7]:
            img = image.load_img(ela_path)
            img = img.resize(model.input_shape[1:-1], Image.LANCZOS)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predictions.append(prediction[0][0]*100)
        os.remove(file_path)
        os.remove(ela_path)
        maximum = np.max(predictions)
        print(predictions)
        return jsonify({'score': "{:.2f}".format(maximum)})
    except Exception as e:
      return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)