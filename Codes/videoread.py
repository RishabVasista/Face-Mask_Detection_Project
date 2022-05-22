import numpy as np
import cv2

# Capture video from file
cap = cv2.VideoCapture('C:/Users/Vasista/Downloads/Scam 1992 The Harshad Mehta Story Ep.05 To 10 (2020) Hindi Web Series HEVC 480p.mkv')

while True:

    ret, frame = cap.read()

    if ret == True:

        cv2.imshow('frame',frame)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()