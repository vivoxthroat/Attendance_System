import cv2
import os

student_name = input("Enter student name: ")

save_path = f"dataset/{student_name}"
os.makedirs(save_path, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print("Move your head slowly. Press ESC to stop.")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))

        if count < 50:
            cv2.imwrite(
                os.path.join(save_path, f"{count}.jpg"),
                face
            )
            count += 1

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Images: {count}/50",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1)

    if key == 27 or count >= 50:
        break

camera.release()
cv2.destroyAllWindows()

print("Dataset collection completed.")