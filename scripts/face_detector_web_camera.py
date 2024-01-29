"""FACE DETECTOR THROUGH WEB CAMERA"""

import cv2

video = cv2.VideoCapture(0)

file = '../files/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(file)

while True:

    ret, frame = video.read()
    if not ret:
        break

    results = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()