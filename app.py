from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import logging
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load the model
model = load_model('handwriting_classification_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to 32x32 pixels
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

@app.route('/')
def index():
    return "Flask app is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image and make prediction
        preprocessed_img = preprocess_image(img)
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map the predicted class to its label
        class_labels = {0: 'Normal', 1: 'Reversal', 2: 'Corrected'}
        result = class_labels[predicted_class]

        # Determine if the child is at risk of dyslexia
        if result == 'Reversal' or result == 'Corrected':
            risk = 'At risk of dyslexia'
        else:
            risk = 'Not at risk of dyslexia'

        return jsonify({'prediction': result, 'risk': risk})
    except Exception as e:
        app.logger.error(f"Error processing the image: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
