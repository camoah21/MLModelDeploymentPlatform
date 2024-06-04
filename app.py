from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import joblib
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'models/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for simplicity
models = {}

@app.route('/upload', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    file = request.files['model']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load the model to ensure it's valid
    model = joblib.load(filepath)
    models[filename] = model
    
    return jsonify({"message": "Model uploaded successfully", "model_name": filename}), 201

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'model_name' not in data or 'input' not in data:
        return jsonify({"error": "Model name and input data are required"}), 400
    
    model_name = data['model_name']
    if model_name not in models:
        return jsonify({"error": "Model not found"}), 404
    
    model = models[model_name]
    input_data = pd.DataFrame(data['input'])
    prediction = model.predict(input_data)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

