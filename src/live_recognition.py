import cv2
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from attendance_logger import mark_attendance
from datetime import datetime

# Load trained model
model = load_model("models/face_model.h5")

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Webcam
camera = cv2.VideoCapture(0)

# To avoid duplicate attendance entries
marked_students = set()

start_time = datetime.strptime("09:30", "%H:%M").time()
end_time = datetime.strptime("10:00", "%H:%M").time()

while True:

    ret, frame = camera.read()

    if not ret:
        break

    current_time = datetime.now().time()
    

    if not (start_time <= current_time <= end_time):

        cv2.putText(
            frame,
            "Attendance Closed",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) == 27:
            break

        continue
    

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Extract face
        face = frame[y:y+h, x:x+w]

        # Resize for model
        face = cv2.resize(face, (224, 224))

        # Normalize
        face = face / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Predict
        prediction = model.predict(face, verbose=0)

        class_id = np.argmax(prediction)

        name = encoder.inverse_transform([class_id])[0]

        confidence = np.max(prediction) * 100

        # Mark attendance once
        if confidence > 90:

            if name not in marked_students:

               mark_attendance(
                name,
                "Neutral"
            )


            marked_students.add(name)

            print(f"{name} marked present")

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Show prediction
        emotion = detect_emotion(face)

        cv2.putText(
            frame,
            f"{name} ({confidence:.1f}%)"
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()