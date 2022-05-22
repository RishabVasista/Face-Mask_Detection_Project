import os

def generate_positive_description_file():
  with open ('pos.txt','w') as f:
      for filename in os.listdir('resized1'):
          f.write('resized1/' + filename + ' 1 0 0 100 100\n')

generate_positive_description_file()

# C:/Users/Vasista/Desktop/project/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec pos.vec  -bg neg.txt -w 24 -h 24 -numPos 4790 -numNeg 18000 -numStages 14 -weightTrimRate 0.95 -mode ALL -precalcValBufSize 3000 -precalcIdxBufSize 3000