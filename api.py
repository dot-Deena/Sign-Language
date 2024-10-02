from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('/Users/deena/Desktop/py/SIgnlanguage/signlanguagedetectionmodel48x48.h5')

# Define your labels
labels = ['A', 'B', 'C', 'D']  # Replace with your actual labels

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame file provided'}), 400

    # Load image file
    frame_file = request.files['frame'].read()
    img = Image.open(io.BytesIO(frame_file))

    # Preprocess the image
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to the size expected by your model
    img = np.array(img).reshape(1, 48, 48, 1) / 255.0

    # Make prediction
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions[0])
    label = labels[predicted_label]

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
