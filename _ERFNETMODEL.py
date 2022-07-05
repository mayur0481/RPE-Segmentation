import tensorflow as tf
from tensorflow import keras
from _CustomLayers import DownsamplerBlock, NonBT1D

class ERFNET:
    @staticmethod
    def build_model(input_shape, dropout = 0.3, activation = 'relu', classes = 2):
        inp = keras.layers.Input(shape = input_shape)
        dsb = DownsamplerBlock(13, activation, name = 'downsample_block_1')(inp)
        dsb1 = DownsamplerBlock(48, activation, name = 'downsample_block_2')(dsb)
        nb1 = NonBT1D(activation = activation)(dsb1)
        nb2 = NonBT1D(activation = activation)(nb1)
        nb3 = NonBT1D(activation = activation)(nb2)
        nb4 = NonBT1D(activation = activation)(nb3)
        nb5 = NonBT1D(activation = activation)(nb4)
        dsb2 = DownsamplerBlock(64, activation = activation)(nb5)
        nb6 = NonBT1D((2,2), activation)(dsb2)
        nb7 = NonBT1D((4, 4), activation)(nb6)
        nb8 = NonBT1D((8, 8), activation)(nb7)
        nb9 = NonBT1D((16, 16), activation)(nb8)
        nb10 = NonBT1D((2, 2), activation)(nb9)
        nb11 = NonBT1D((4, 4), activation)(nb10)
        nb12 = NonBT1D((8, 8), activation)(nb11)
        nb13 = NonBT1D((16, 16), activation)(nb12)
        dconv1 = keras.layers.Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same', activation = activation)(nb13)
        nb14 = NonBT1D(activation = activation)(dconv1)
        nb15 = NonBT1D(activation = activation)(nb14)
        dconv2 = keras.layers.Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same', activation = activation)(nb15)
        nb16 = NonBT1D(activation = activation)(dconv2)
        nb17 = NonBT1D(activation = activation)(nb16)
        if classes == 2:
            dconv3 = keras.layers.Conv2DTranspose(1, (3, 3), strides = (2, 2), padding = 'same', activation = 'sigmoid')(nb17)
        else:
            dconv3 = keras.layers.Conv2DTranspose(classes, (3, 3), strides = (2, 2), padding = 'same', activation = 'softmax')(nb17)
            
        model = keras.models.Model(inputs = [inp], outputs = [dconv3])
        return model
    
    
    
