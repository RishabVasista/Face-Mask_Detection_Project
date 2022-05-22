#rescaling
import cv2 as cv
#rescaling images
img=cv.imread('C:/Users/Vasista/Desktop/project/openCV tutorial files/Photos/pic1.jpg')
cv.imshow('picture',img)
def rescaleFrame(frame,scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)

    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


img1=rescaleFrame(img)
cv.imshow('rescaled picture',img1)
cv.waitKey(0)

#rescaling videos
cap=cv.VideoCapture('C:/Users/Vasista/Downloads/Scam 1992 The Harshad Mehta Story Ep.05 To 10 (2020) Hindi Web Series HEVC 480p.mkv')
while True:

    ret, frame=cap.read()
    frame_resized=rescaleFrame(frame,0.2)
    if ret==True:
        cv.imshow('your name',frame)
        cv.imshow('your name1',frame_resized)
    if cv.waitKey(30) & 0xFF==ord('d'):
         break            
    

cap.release()
cv.destroyAllWindows()
