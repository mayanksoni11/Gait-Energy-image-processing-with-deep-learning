import numpy as np
import glob, pickle
import os, sys
import argparse
import cv2
import keras
from keras.layers import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras.layers import Reshape
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adagrad
import PIL
import os
from PIL import Image
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import math
K.set_image_dim_ordering('th') 

img_rows = 256
img_cols = 256
SHAPE = 256
BATCH = 4
IN_CH = 3
OUT_CH = 3
LAMBDA = 100
NF = 64 # number of filter
BATCH_SIZE = 10

def generator_model():
    global BATCH_SIZE
    # imgs: input: 256x256xch
    # U-Net structure, must change to relu
    inputs = Input((IN_CH, img_cols, img_rows))
    
    e1 = BatchNormalization(mode=0)(inputs)
    
    e1 = Convolution2D(64, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e1)
    e1 = BatchNormalization(mode=0)(e1)
    e2 = Convolution2D(128, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e1)
    e2 = BatchNormalization(mode=0)(e2)

    e3 = Convolution2D(256, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e2)
    e3 = BatchNormalization(mode=0)(e3)    
    e4 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e3)
    e4 = BatchNormalization(mode=0)(e4)

    e5 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e4)
    e5 = BatchNormalization(mode=0)(e5)
    e6 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e5)
    e6 = BatchNormalization(mode=0)(e6)  

    e7 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e6)
    e7 = BatchNormalization(mode=0)(e7)
    e8 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e7)
    e8 = BatchNormalization(mode=0)(e8)
    
    d1 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 2, 2), border_mode='same')(e8)
    d1 = keras.layers.Concatenate(axis=1)([d1, e7])
    d1 = BatchNormalization(mode=0)(d1)
    
    d2 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 4, 4), border_mode='same')(d1)
    d2 = keras.layers.Concatenate(axis=1)([d2, e6])
    d2 = BatchNormalization(mode=0)(d2)
    
    d3 = Dropout(0.2)(d2)
    d3 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 8, 8), border_mode='same')(d3)
    d3 = keras.layers.Concatenate(axis=1)([d3, e5])
    d3 = BatchNormalization(mode=0)(d3)    

    d4 = Dropout(0.2)(d3)
    d4 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 16, 16), border_mode='same')(d4)
    d4 = keras.layers.Concatenate(axis=1)([d4, e4])
    d4 = BatchNormalization(mode=0)(d4)    
    
    d5 = Dropout(0.2)(d4)
    d5 = Deconvolution2D(256, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 256, 32, 32), border_mode='same')(d5) 
    d5 = keras.layers.Concatenate(axis=1)([d5, e3])
    d5 = BatchNormalization(mode=0)(d5)
    
    d6 = Dropout(0.2)(d5)
    d6 = Deconvolution2D(128, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 128, 64, 64), border_mode='same')(d6)
    d6 = keras.layers.Concatenate(axis=1)([d6, e2])    
    d6 = BatchNormalization(mode=0)(d6)   
    
    d7 = Dropout(0.2)(d6)
    d7 = Deconvolution2D(64, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 64,128, 128), border_mode='same')(d7)        
    d7 = keras.layers.Concatenate(axis=1)([d7, e1])    
    
    d7 = BatchNormalization(mode=0)(d7)
    d8 = Deconvolution2D(3, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 3, 256, 256), border_mode='same')(d7)
    
    d8 = BatchNormalization(mode=0)(d8)
    d9 = Activation('tanh')(d8)
    
    model = Model(input=inputs, output=d9)
    
    return model
def discriminator_model():
    """ return a (b, 1) logits"""

    model = Sequential()
    model.add(Convolution2D(64, 4, 4,border_mode='same',input_shape=(IN_CH, img_cols, img_rows)))
    model.add(BatchNormalization(mode=0))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 4, 4,border_mode='same'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation('tanh'))
    model.add(Convolution2D(1, 4, 4,border_mode='same'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation('tanh'))
    
    model.add(Activation('sigmoid'))
    return model
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((3,height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[:,i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, :]
    return image

def generator_containing_discriminator(generator, discriminator):
    inputs = Input((IN_CH, img_cols, img_rows))
    x_generator = generator(inputs)
    
    merged = concatenate([inputs, x_generator], axis=-1)
    
    discriminator.trainable = False
    
    x_discriminator = discriminator(merged)
    
    
    model = Model(input=inputs, output=[x_generator,x_discriminator])
    
    return model

def discriminator_loss(y_true,y_pred):
    BATCH_SIZE=10
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.concatenate([K.ones_like(K.flatten(y_pred[:BATCH_SIZE,:,:,:])),K.zeros_like(K.flatten(y_pred[:BATCH_SIZE,:,:,:])) ]) ), axis=-1)

def discriminator_on_generator_loss(y_true,y_pred):
    BATCH_SIZE=10
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.ones_like(K.flatten(y_pred))), axis=-1)

def generator_l1_loss(y_true,y_pred):
    BATCH_SIZE=10
    return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)

def train(BATCH_SIZE):
    path1 = r'E:\\CASIA_B90PerfectCentrallyAlinged_EnergyImage\\'
    train_imgs = []
    train_labels = []
    subjects = os.listdir(path1)
    numberOfSubject = len(subjects)
    print('Number of Subjects: ', numberOfSubject)
    for number1 in range(0, numberOfSubject):  # numberOfSubject
        path2 = (path1 + subjects[number1] + '\\')
        sequences = os.listdir(path2);
        for number2 in range(0, 2):
            path3 = path2 + sequences[number2]
#            print(path3)
            img = cv2.imread(path3 , 0)
            img = cv2.resize(img,(256,256))
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            train_imgs.append(img)
            label = [0] * numberOfSubject
            label[number1] = 1
            train_labels.append(label)
                
    X_train = np.array(train_imgs)
    Y_train = np.array(train_labels)

    #print(np.shape(X_train))
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    Y_train = (Y_train.astype(np.float32) - 127.5)/127.5
    #X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    #Y_train = Y_train.reshape((Y_train.shape[0], 1) + Y_train.shape[1:])
    X_train = X_train.reshape(X_train.shape[0], 3, 256, 256)

    discriminator = discriminator_model()
    generator = generator_model()
    generator.summary()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    d_optim = Adagrad(lr=0.005)
    g_optim = Adagrad(lr=0.005)
    generator.compile(loss='mse', optimizer="rmsprop")
    discriminator_on_generator.compile(loss=[generator_l1_loss,discriminator_on_generator_loss] , optimizer="rmsprop")
    discriminator.trainable = True
    discriminator.compile(loss=discriminator_loss, optimizer="rmsprop")

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            image_batch = Y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
           
            generated_images = generator.predict(X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE])
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                image = np.swapaxes(image,0,2)
                cv2.imwrite(str(epoch)+"_"+str(index)+".png",image)      
                #Image.fromarray(image.astype(np.uint8)).save(str(epoch)+"_"+str(index)+".png")

            real_pairs = np.concatenate((X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE,:,:,:],image_batch),axis=1) 
            fake_pairs = np.concatenate((X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE,:,:,:],generated_images),axis=1)
            X = np.concatenate((real_pairs,fake_pairs))
            y = np.zeros((20,1,64,64)) #[1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            pred_temp = discriminator.predict(X)
            #print(np.shape(pred_temp))
            print("batch %d d_loss : %f" % (index, d_loss))
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE,:,:,:], [image_batch,np.ones((10,1,64,64))] )
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss[1]))
            if index % 20 == 0:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    path1 = r'E:\\CASIA_B90PerfectCentrallyAlinged_EnergyImage\\'
    train_imgs = []
    train_labels = []    
    subjects = os.listdir(path1)
    numberOfSubject = len(subjects)
    print('Number of Subjects: ', numberOfSubject)
    for number1 in range(0, numberOfSubject):  # numberOfSubject
        path2 = (path1 + subjects[number1] + '\\')
        sequences = os.listdir(path2);
        numberOfsequences = len(sequences)
        for number2 in range(4, numberOfsequences):
            path3 = path2 + sequences[number2]
#            print(path3)
            img = cv2.imread(path3 , 0)
            img = cv2.resize(img,(256,256))
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            train_imgs.append(img)
            label = [0] * numberOfSubject
            label[number1] = 1
            train_labels.append(label)
                
    X_train = np.array(train_imgs)
    Y_train = np.array(train_labels)
    #print(net_data('test')p.shape(X_train))
    X_train = X_train.reshape(X_train.shape[0], 3, 256, 256)
    print(np.shape(X_train))
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    Y_train = (Y_train.astype(np.float32) - 127.5)/127.5
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    
    
    #aj=generator.get_weights()
    generator.load_weights('generator')
    
    
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        
        generated_images = generator.predict(X_train, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) + (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        
        generated_images = generator.predict(X_train)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    image = np.swapaxes(image,0,2)
    cv2.imwrite('generated.png',image) 
    
def split_input(img):
    """
    img: an 512x256x3 image
    :return: [input, output]
    """
    input, output = img[:,:img_cols,:], img[:,img_cols:,:]

    if args.mode == 'BtoA':
        input, output = output, input
    if IN_CH == 1:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    if OUT_CH == 1:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    return [input, output]

#def get_data(datadir):
#    #datadir = args.data
#    # assume each image is 512x256 split to left and right
#    imgs = glob.glob(os.path.join(datadir, '*.jpg'))
#    data_X = np.zeros((len(imgs),3,img_cols,img_rows))
#    data_Y = np.zeros((len(imgs),3,img_cols,img_rows))	
#    i = 0
#    for file in imgs:
#        img = cv2.imread(file,cv2.IMREAD_COLOR)
#        img = cv2.resize(img, (img_cols*2, img_rows)) 
#        #print('{} {},{}'.format(i,np.shape(img)[0],np.shape(img)[1]))
#        img = np.swapaxes(img,0,2)
#
#        X, Y = split_input(img)
#
#        data_X[i,:,:,:] = X
#        data_Y[i,:,:,:] = Y
#        i = i+1
#    return data_X, data_Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='A directory of 512x256 images')
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    global args
    args = parser.parse_args()
    gen = generator_model()
    gen.compile(loss='binary_crossentropy', optimizer="SGD")
    out = gen.predict(np.zeros((10,3,256,256)))
    train(10)
    generate(10)