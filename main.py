from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms

import os
import numpy as np
import warnings
import torch
import gdown
# Initialize Flask app

warnings.filterwarnings("ignore");

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Ensure the uploads directory exists
upload_dir = "./uploads"
os.makedirs(upload_dir, exist_ok=True)

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),          # Convert to PyTorch Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize if needed
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
model_url = "https://drive.google.com/uc?id=1-KQz79KoNIjuNveqyD-eh137KfesrVsG"
model_path = "./model/ela_model7_resnet18_20epochs.keras"
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
model_tf = load_model(model_path)
# model4= load_model("./model/ela_model4_xception_32epochs.keras")
# model5= load_model("./model/ela_model5_xception_attentnion_18epochs.keras")
# model7= load_model("./model/ela_model7_resnet18_20epochs.keras")
tensor_models = [model_tf]
# Load the PyTorch model
# torch1_path = "./model/resnet18_2_ela_model.pth"
# torch1 = torch.load(torch1_path, weights_only = False, map_location=torch.device('cpu'))

# torch2_path = "./model/Resnet18_30ep_model.pth"
# torch2 = torch.load(torch2_path, weights_only = False, map_location=torch.device('cpu'))

# torch3_path = "./model/Resnet50_model.pth"
# torch3 = torch.load(torch3_path, weights_only = False, map_location=torch.device('cpu'))
# torch_models = [torch1]

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
        for tf_model in tensor_models:
            img = image.load_img(ela_path)
            img = img.resize(tf_model.input_shape[1:-1], Image.LANCZOS)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = tf_model.predict(img_array)
            tf_pred = prediction[0][0]*100
            print('TF Prediction:', tf_pred)
            predictions.append(tf_pred)

        # for torch_model in torch_models:
        #     img = image.load_img(ela_path)
        #     img_tensor = transform(img).unsqueeze(0)
        #     torch_model.eval()
        #     with torch.no_grad():
        #         output = torch_model(img_tensor.to(device))  # Predict
        #         prob = torch.softmax(output, dim=1)  # Convert to probabilities
        #         pred_class = torch.argmax(prob, dim=1).item()  # Get the predicted class
        #         torch_pred = prob[0][1].item()*100
        #         print('PyTorch Prediction:', torch_pred)
        #         predictions.append(torch_pred)
        maximum = np.max(predictions)
        os.remove(file_path)
        os.remove(ela_path)
        return jsonify({'score': "{:.2f}".format(maximum)})
    except Exception as e:
      return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
    # app.run(debug=True, host='0.0.0.0', port=5001)