import os

def generate_negative_description_file():
  with open ('neg.txt','w') as f:
      for filename in os.listdir('resized2'):
          f.write('resized2/' + filename + '\n')

generate_negative_description_file()
#C:/Users/Vasista/Desktop/project/opencv/build/x64/vc15/bin/opencv_annotation.exe --annotations=pos.txt --images=positive/
# C:/Users/Vasista/Desktop/project/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

#C:/Users/Vasista/Desktop/project/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec pos.vec  -bg neg.txt -w 24 -h 24 -numPos 90 -numNeg 180 -numStages 10