import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize

data_dir = 'dataset/PlantVillage'
categories = os.listdir(data_dir)

X = []
y = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = imread(img_path)
            img_resized = resize(img, (64, 64), anti_aliasing=True)
            X.append(img_resized.flatten())
            y.append(category)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

os.makedirs("model", exist_ok=True)
with open("model/potato_model.pkl", "wb") as f:
    pickle.dump((model, le), f)