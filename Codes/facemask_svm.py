#Code to implement multiple classifiers under one program.

#importing directories
import cv2 as cv
import pickle
from svm_dtree import find_masks

#preloading all parameters
bw_threshold = 80
font = cv.FONT_HERSHEY_PLAIN
org = (30, 30)
worn_mask_font_color = (255, 255, 255)
not_worn_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
worn_mask = "Thank You for wearing a MASK"
not_worn_mask = "Please wear a MASK"
filename = 'facemask_model_svm.sav'
filename1 = 'XScaler_model_svm.sav'

svm_facemask=pickle.load(open(filename, 'rb'))
X_scaler1=pickle.load(open(filename1,'rb'))

#loading classifiers
face_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_frontalface.xml')
mouth_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_mcs_mouth1.xml')
nose_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_nose1.xml')
facemask_cascade =cv.CascadeClassifier('test5/cascade.xml')
mask_cascade= cv.CascadeClassifier('test3.xml')

#taking real time input
#live camera feed--> 0 means main camera 
cap = cv.VideoCapture(0)

while 1:
    # Get individual frame
    ret, img = cap.read()
    #Get non mirrored feed
    img = cv.flip(img,1)

    # Convert Image into gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image in black and white     
    #performing closing operation
    #dialated = cv.dilate(gray,(3,3),iterations=3)     
    #eroded = cv.erode(dialated,(3,3),iterations=3)
    #black_and_white = cv.threshold(dialated,bw_threshold,255,cv.THRESH_OTSU)                                                                                                       
    #cv.imshow('black_and_white', black_and_white)

    # detect face & facemask
    faces = face_cascade.detectMultiScale(gray, 1.05, 3)
    #facemasks = facemask_cascade.detectMultiScale(gray,1.1,4)
    facemasks = find_masks(img,'BGR2HSV',svm_facemask,X_scaler1,9,8,2)
    # Face prediction for black and white
    #facemasks_bw = facemask_cascade.detectMultiScale(eroded,1.1,4)


    #if(len(faces) == 0 and len(faces_bw) == 0 and len(facemasks) == 0 and len(facemasks_bw)== 0):
    if(len(faces) == 0 and len(facemasks) == 0):
        cv.putText(img, "No face found...", org, font, font_scale, worn_mask_font_color, thickness, cv.LINE_AA)
    elif(len(faces) == 0 ):
        cv.putText(img, worn_mask, org, font, font_scale, worn_mask_font_color, thickness, cv.LINE_AA)
    else:
        # Draw rectangle on Face
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]


            # Detect lips
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
            
           # Face detected but Lips not detected which means person is wearing mask
        if(len(mouth_rects) == 0 and len(facemasks) == 1):
            cv.putText(img, worn_mask, org, font, font_scale, worn_mask_font_color, thickness, cv.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                    # person is not waring mask
                    cv.putText(img, not_worn_mask, org, font, font_scale, not_worn_mask_font_color, thickness, cv.LINE_AA)

                    #cv.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break
    # Show frame with results
    cv.imshow('Mask Detection', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break