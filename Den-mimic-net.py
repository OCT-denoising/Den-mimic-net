"""

@author: Mahnoosh Tajmirriahi
"""
from datagen import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Dropout
from keras.models import Model
import numpy as np
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import *
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from skimage.util import img_as_float
######## Autoencoder network###########################################
##Encoder part
input_img = Input(shape=(256, 256, 1))  
x = Conv2D(16, (3, 3) ,padding='same',name='layer_one')(input_img)
x=LeakyReLU(alpha=0.4)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3),  padding='same',name='layer_two')(x)
x=LeakyReLU(alpha=1)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
##bottleneck
x = Conv2D(4, (3, 3),  padding='same',name='layer_three')(x)
x=LeakyReLU(alpha=0.5)(x)
##Decoder part
x = Conv2D(8, (3, 3), padding='same',name='layer_four')(x)
x=LeakyReLU(alpha=1)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), padding='same',name='layer_five')(x)
x=LeakyReLU(alpha=0.4)(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), padding='same',name='layer_six')(x)
decoded=LeakyReLU(alpha=0.3)(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

######### lighter network###############################################
'''
This artitechture can be used for lower amount of training data.
'''
# input_img = Input(shape=(256, 256, 1))  
# x = Conv2D(16, (3, 3) ,padding='same',name='layer_one')(input_img)
# x=LeakyReLU(alpha=0.4)(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(16, (3, 3), padding='same',name='layer_two')(x)
# x=LeakyReLU(alpha=0.4)(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), padding='same',name='layer_three')(x)
# decoded=LeakyReLU(alpha=0)(decoded)
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='mse')
#########Add Augmentation################################################
data_gen_args = dict(rotation_range=10,featurewise_center=False,
    featurewise_std_normalization=False,
                    width_shift_range=0.08,
                    height_shift_range=0.08,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=True,
                    fill_mode='nearest')
# data_gen_args=dict() # uncomment this line for not using augmentation

##########Training######################################################
myGene =trainGenerator(2,'/content/drive/','noisy_den_gauss','amini_den_gauss',data_gen_args)#specify the noisy and clean image path
history=autoencoder.fit_generator(myGene,steps_per_epoch=50,epochs=101,shuffle=True)

########## training with Validation#####################################
#myGene =trainGenerator(2,'/content/drive/','noisy_den_gauss','amini_den_gauss',data_gen_args)
# valGene=validateGenerator(2,'/content/drive/','valimage','denvalimage',data_gen_args)
# history=autoencoder.fit_generator(myGene,steps_per_epoch=60,validation_data=valGene,validation_steps=19,epochs=100,shuffle=True)
########## predict test results#########################################

testGene = testGenerator("/content/drive/testn/") #specify the test image path
results =img_as_float (autoencoder.predict_generator(testGene,20,verbose=1))

######### save results and weights#######################################
saveResult("/content/drive/resultscrop/",results) # specify the path results must be saved
autoencoder.save_weights('/content/drive/den_mimic/test1_weights.h5')# specify the path weights must be saved

