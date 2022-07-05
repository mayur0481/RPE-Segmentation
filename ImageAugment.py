import tensorflow as tf

def _augment(hfp = 0.5, vfp = 0.1):
    def _augment_fn(img, label):
        flip_prob = tf.random.uniform(())
        img, label = tf.cond(tf.less(flip_prob, hfp), lambda : (tf.image.flip_left_right(img), tf.image.flip_left_right(label)), lambda:(img, label))
        
        flip_prob = tf.random.uniform(())
        img, label = tf.cond(tf.less(flip_prob, vfp), lambda : (tf.image.flip_up_down(img), tf.image.flip_up_down(label)), lambda : (img, label))


        return img, label


    return _augment_fn



        
