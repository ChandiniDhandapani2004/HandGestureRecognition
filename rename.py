#This code is used to rename the files and change their extensions if needed.
#This code can rename multiple image files at a time.

import os
os.chdir('E:\\novitech internship\\Hand Gesture Recognition\\images\\Polar_bear')
i=1
for file in os.listdir():
    src=file
    dst="jumanji"+str(i)+".png"
    os.rename(src,dst)
    i+=1
