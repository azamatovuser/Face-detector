"""FACE DETECTOR THROUGH VIDEO"""

import cv2

video = cv2.VideoCapture('../videos/cosmos.mp4')

file = '../files/haarcascade_frontalface_default.xml'
faces = cv2.CascadeClassifier(file)

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = faces.detectMultiScale(frame, scaleFactor=2, minNeighbors=3)

    for (x, y, w, h) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()