import tensorflow as tf
import keras
import cv2
import numpy as np
from keras.models import load_model
import matplotlib
import os
path1 = r'E:\model gen\\'
count=0;
files = os.listdir(path1)
for names in files:
    print(names)
    generator = load_model(path1+names)
    path = "E:/CASIA_B90PerfectCentrallyAlinged_EnergyImage/049/bg-02.png"
    img=cv2.imread(path)
    img = cv2.resize(img,(256,256))
    print(img.shape)
    img=np.expand_dims(img,axis=0)
    print(img.shape)
    img1=generator.predict(img)
    
    
    
    img_1 = np.reshape(img1 , (256,256,3))
    print(img1.shape)
    
    
    
    matplotlib.image.imsave(r'E:\result\result_%d.png'%count,img_1)
    count=count+1
