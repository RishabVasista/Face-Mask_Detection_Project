import cv2 as cv

aa=cv.imread('C:/Users/Vasista/Desktop/project/images/maksssksksss111.png')
bb=cv.imread('C:/Users/Vasista/Desktop/project/images/maksssksksss118.png')
cascade=cv.CascadeClassifier('haar_1200.xml')
faces=cascade.detectMultiScale(aa,1.3,4)
for (x,y,w,h) in faces:
            cv.rectangle(aa, (x, y), (x + w, y + h), (255, 255, 255), 1) 
cv.imwrite("Image1.png",aa)
faces=cascade.detectMultiScale(bb,1.3,4)
for (x,y,w,h) in faces:
            cv.rectangle(bb, (x, y), (x + w, y + h), (255, 255, 255), 1) 
cv.imwrite("Image2.png",bb)
exit