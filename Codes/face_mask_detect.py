import cv2 as cv

# Load the cascade
face_cascade = cv.CascadeClassifier('C:/Users/Vasista/Desktop/project/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    img = cap.read()
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1 ,2)
   # faces_bw= face_cascade.detectMultiScale(black_and_white, 1.1 ,4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
     cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

    # Display
    cv.imshow('img', img)

    # Stop if escape key is pressed
    k = cv.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()
cv.destroyAllWindows()
