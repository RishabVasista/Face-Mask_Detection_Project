import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_frontalface_default.xml')
mouth_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_mcs_mouth.xml')




# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK"

# Read video
cap = cv.VideoCapture(0)

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv.flip(img,1)

    # Convert Image into gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image in black and white
    black_and_white = cv.threshold(gray,90 ,255 , cv.THRESH_BINARY)

    #cv.imshow('black_and_white', black_and_white)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)


    if(len(faces) == 0 and len(faces_bw) == 0):
        cv.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv.LINE_AA)
    else:
        # Draw rectangle on Face
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]


            # Detect lips
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
            
           # Face detected but Lips not detected which means person is wearing mask
        if(len(mouth_rects) == 0):
            cv.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                    # person is not waring mask
                    cv.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv.LINE_AA)

                    #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break
    # Show frame with results
    cv.imshow('Mask Detection', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv.destroyAllWindows()