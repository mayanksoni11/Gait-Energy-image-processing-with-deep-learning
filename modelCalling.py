import tensorflow as tf
import keras
import cv2
import numpy as np
from keras.models import load_model
import matplotlib
import os
path1 = r'E:\model gen\\'
count=50;
files = os.listdir(path1)
for names in files:
    print(names)
    generator = load_model(path1+names)
    path = r'E:\GaitDatasetB-silh_PerfectlyAlinged_max_PEI_Images_with_Ref_Poses3\001\\cl-012.png'
    img=cv2.imread(path)
    print(img.shape)
    img = cv2.resize(img,(256,256))
#    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    print(img.shape)
    img = np.array(img)/255.0
    img=np.expand_dims(img,axis=0)
    print(img.shape)
    img1=generator.predict(img)
    
    
    
    img_1 = np.reshape(img1 , (256,256,3))
    print(img1.shape)
    
    
    
    matplotlib.image.imsave(r'E:\result\result_%d.png'%count,img_1)
    count=count+1
