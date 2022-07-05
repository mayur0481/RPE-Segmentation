import tensorflow as tf
from tensorflow import keras


class DiceLoss(keras.losses.Loss):
    def __init__(self, epsilon = 1e-6, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        numerator = 2*tf.reduce_sum(y_true * y_pred, axis = [1, 2, 3])
        denominator = tf.reduce_sum(y_true + y_pred, axis = [1, 2, 3])

        return 1 - (numerator + self.epsilon)/(denominator + self.epsilon)

    def get_config(self):
        base_config = super().get_config()
        {**base_config, 'epsilon':self.epsilon}



class BCELoss(keras.losses.Loss):
    def __init__(self, lamda = 1, **kwargs):
        self.lamda = lamda
        self.ce = keras.losses.BinaryCrossentropy()
        self.dl = DiceLoss()
        super().__init__(**kwargs)


    def call(self, y_true, y_pred):
        return self.ce(y_true, y_pred) + self.dl(y_true, y_pred) * self.lamda

    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'lamda':self.lamda}

