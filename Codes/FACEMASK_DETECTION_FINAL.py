#importing directories
import cv2 as cv

#loading classifiers
mouth_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/THE SYSTEM/STUDY/project/haarcascade_mcs_mouth1.xml')
nose_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/THE SYSTEM/STUDY/project/haarcascade_mcs_nose1.xml')
facemask_cascade =cv.CascadeClassifier('LBP1200.xml')
#face_cascade=cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_frontalface.xml')
face_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/THE SYSTEM/STUDY/project/haarcascade_mcs_face.xml')

#taking real time input
#live camera feed--> 0 means main camera 
cap = cv.VideoCapture(0)

while 1 :
    # Get individual frame
    ret, img = cap.read()
    #Get non mirrored feed
    img = cv.flip(img,1)

    # Convert Image into gray
    #img = cv.erode(img,(3,3),iterations=3)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detect upper body using classifier
    #faces = face_cascade.detectMultiScale(gray,1.4,3)
    faces = face_cascade.detectMultiScale(gray,1.1,3)
    if len(faces) == 0:
        cv.putText(img,"No face detected",(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv.LINE_4)
    else:
        for (x,y,w,h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)    
         #detect facemask
            crop = gray[x:x+w,y:y+h]
            facemasks = facemask_cascade.detectMultiScale(crop,1.2,3)
            mouth = mouth_cascade.detectMultiScale(crop,1.05,3)
            nose = nose_cascade.detectMultiScale(crop,1.02,3)
            if (len(nose) == 1 or len(mouth)==1) and len(facemasks) == 1:
                cv.putText(img,"Please wear the face mask properly.",(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv.LINE_4)
            elif len(facemasks)==1 or len(nose) == 0 or len(mouth) == 0 :
                cv.putText(img,"Thank you for wearing the face mask properly.",(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv.LINE_4)
            else :cv.putText(img,"No Mask Detected.",(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv.LINE_4)
    cv.imshow('Mask Detection', img)
    k = cv.waitKey(10) & 0xff
    if k == 27:
        break
 