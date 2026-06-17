import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pickle

DATASET = "dataset"

images = []
labels = []

for person in os.listdir(DATASET):

    person_folder = os.path.join(DATASET, person)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        images.append(img)
        labels.append(person)

X = np.array(images)
y = np.array(labels)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(
    y.shape[1],
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=8
)

model.save("models/face_model.h5")

print("Model saved successfully.")