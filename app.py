from flask import Flask, render_template, request
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open("model/potato_model.pkl", "rb") as f:
    model, le = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = imread(file_path)
    img_resized = resize(img, (64, 64), anti_aliasing=True).flatten().reshape(1, -1)

    prediction = model.predict(img_resized)
    label = le.inverse_transform(prediction)[0]

    return render_template("result.html", disease=label, image_path=file_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)