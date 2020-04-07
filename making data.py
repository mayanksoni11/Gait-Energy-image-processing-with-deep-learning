import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

path1 = r'E:\\GaitDatasetB-silh_PerfectlyAlinged_max_PEI_Images_with_Ref_Poses3\\'


subjects = os.listdir(path1)
numberOfSubject = len(subjects)
train_imgs = []
train_imgs1 = []
print('Number of Subjects: ', numberOfSubject)
count=0;
for number1 in range(0, 60):  # numberOfSubject
    path2 = (path1 + subjects[number1] + '\\')
    sequences = os.listdir(path2);
    print(path2)
    numberOfsequences = len(sequences)
    for number2 in range(0, 12):
        print(number2)
        path3 = path2 + sequences[number2]
        print(path3)
        img = cv2.imread(path3 , 0)
        img = cv2.resize(img,(256,256))
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
#        count=count+1;
#        img = img_to_array(img)
        train_imgs.append(img)
        
for number1 in range(0, 60):  # numberOfSubject
    path2 = (path1 + subjects[number1] + '\\')
    sequences = os.listdir(path2);
    numberOfsequences = len(sequences)
    for number2 in range(12, 24):
        path3 = path2 + sequences[number2]
        print(path3)
        img = cv2.imread(path3 , 0)
        img = cv2.resize(img,(256,256))
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        count=count+1;
#        img = img_to_array(img)
        train_imgs1.append(img)


print(train_imgs1[1].shape)
for x in range(0, count):
     
     vis = np.concatenate((train_imgs1[x], train_imgs[x]), axis=1)
     cv2.imwrite(r'E:\nayadata\trainPOSE\dataPose_%d.jpg'%x,vis)
 