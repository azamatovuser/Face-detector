"""FACE DETECTOR THROUGH IMAGE"""

import cv2

image = cv2.imread('../images/people.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

file = '../files/haarcascade_frontalface_default.xml'
faces = cv2.CascadeClassifier(file)


results = faces.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=3)

for (x, y, w, h) in results:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

cv2.imshow('result', image)
cv2.waitKey(0)