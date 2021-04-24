"""

@author: Mahnoosh Tajmirriahi
"""

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
import time
import os
import skimage.io as io
import skimage.transform as trans

def testGenerator(test_path,num_image =20,target_size = (256,256),flag_multi_class = False,image_color_mode = "grayscale"):
    import imageio
    for i in range(num_image):
        img = io.imread((test_path+"%d.png"%(i)))
        img = img /255
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
        

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    from skimage.util import img_as_float
    for i,item in enumerate(npyfile):
        img =item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_float(img))
        
data_gen_args=dict()
###############lighter network#################################
input_img = Input(shape=(256, 256, 1)) ##here you can change the size of image to 512x512
x = Conv2D(16, (3, 3) ,padding='same',name='layer_one')(input_img)
x=LeakyReLU(alpha=.4)(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3), padding='same',name='layer_two')(x)
x=LeakyReLU(alpha=.4)(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), padding='same')(x)
decoded=LeakyReLU(alpha=.05)(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
##############################################################
autoencoder.load_weights('/content/drive/den_mimic/test2_weights.h5')##specify the weights path
testGene = testGenerator("/content/drive/testn/") ##specify the test images path
time1=time.time()     
results =img_as_float (autoencoder.predict_generator(testGene,20,verbose=1))
time2=time.time()
print(time2-time1)
saveResult("/content/drive/resultscrop/",results) ## specify the results path