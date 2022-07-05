import tensorflow as tf
from tensorflow import keras

# SCNN LAYERS
class SCNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        c = input_shape[-1]
        self.conv_d = keras.layers.Conv2D(c, (1, 9), padding = 'same', activation = 'relu')
        self.conv_u = keras.layers.Conv2D(c, (1, 9), padding = 'same',  activation = 'relu')
        self.conv_r = keras.layers.Conv2D(c, (9, 1), padding = 'same', activation = 'relu')
        self.conv_l = keras.layers.Conv2D(c, (9, 1), padding = 'same',  activation = 'relu')
        super().build(input_shape)
    
    
    def call(self, x):
        
        z = tf.unstack(x, axis = -3)
        
        for i in range(x.shape[-3]):
            z[i] = tf.expand_dims(z[i], axis = -3)
        
        
        
        for i in range(1, x.shape[-3]):
            z[i] += self.conv_d(z[i-1])
        
        for i in range(x.shape[-3] - 2, 0, -1):
            z[i] += self.conv_u(z[i+1])
        
        
        
        z = tf.concat(z, axis = -3)
        
        z = tf.unstack(z, axis = -2)
        
        for i in range(0, x.shape[-2]):
            z[i] = tf.expand_dims(z[i], axis = -2)
         
        
        
        for i in range(1, x.shape[-2]):
            z[i] += self.conv_l(z[i-1])
            
        for i in range(x.shape[-2] -2, 0, -1):
            z[i] += self.conv_d(z[i+1])

        
        z = tf.concat(z, axis = -2)
        return z


     


# ERFNET LAYERS
    
class DownsamplerBlock(keras.layers.Layer):
    def __init__(self, filters, activation = 'relu', **kwargs):
        self.filters = filters
        self.activation = activation
        self.conv_layer = keras.layers.Conv2D(filters, (3, 3), strides = (2, 2), activation = activation, padding = 'same')
        self.max_pool_layer = keras.layers.MaxPooling2D(2)
        self.concat_layer = keras.layers.Concatenate()
        super().__init__(**kwargs)
    
    def call(self, X):
        A = self.conv_layer(X)
        B = self.max_pool_layer(X)
        return self.concat_layer([A, B])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'filters':self.filters, 'activation':self.activation}
    
    
class NonBT1D(keras.layers.Layer):
    def __init__(self, dilation = (1, 1), activation = 'relu', dropout = 0.1, **kwargs):
        self.dilation = dilation
        self.activation = activation
        self.dropout = dropout
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        #inp_shape = [None, H, W, C]
        _, _, _, C = input_shape
        act = self.activation
        dil = self.dilation
        drop = self.dropout
        self.conv1 = keras.layers.SeparableConv2D(C, (3, 1), padding = 'same', activation = act, dilation_rate = dil)
        self.conv2 = keras.layers.SeparableConv2D(C, (1, 3), padding = 'same', activation = act, dilation_rate = dil)
        self.conv3 = keras.layers.SeparableConv2D(C, (3, 1), padding = 'same', activation = act, dilation_rate = dil)
        self.conv4 = keras.layers.SeparableConv2D(C, (1, 3), padding = 'same', activation = act, dilation_rate = dil)
        self.dropout_layer = keras.layers.Dropout(drop)
        super().build(input_shape)
    
    def call(self, X):
        Z = self.conv1(X)
        Z = self.conv2(Z)
        Z = self.conv3(Z)
        Z = self.conv4(Z)
        Z = Z + X
        Z = self.dropout_layer(Z)
        return Z 

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'dilation':self.dilation, 'activation':self.activation, 'dropout':self.dropout}


    

class BinaryPredictor(keras.layers.Layer):
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
      
    
    def call(self, X):
        X = tf.where(X < self.threshold, 0, 1)
        return X
   
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold':self.threshold}


class MultiClassPredictor(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
    def call(self, X):
        X = tf.argmax(X, -1)
        X = tf.expand_dims(X, -1)
        return X


class CustomAccuracy(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.acc = keras.metrics.Accuracy()
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        # y_pred = [None, h, w, 5]
        # y_true = [None, h, w, 1]
        y_pred = tf.argmax(y_pred, -1)
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        self.acc.update_state(y_true, y_pred)
    
    def result(self):
        return self.acc.result()
    
class CustomMeanIoU(keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.where(y_pred >= 0.5, 1, 0), sample_weight)